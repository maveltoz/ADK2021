import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.core.post_processing import oks_nms
from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

    return bbox_xywh


def _xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0] - 1
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1] - 1

    return bbox_xyxy


def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the img_or_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results


def _inference_single_pose_model(model,
                                 img_or_path,
                                 bboxes,
                                 dataset,
                                 return_heatmap=False):
    """Inference human bounding boxes.

    num_bboxes: N
    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    flip_pairs = None

    if dataset in 'AnimalPoseDataset':
        flip_pairs = [[3, 6], [4, 7], [5, 8], [10, 13], [11, 14], [12, 15]]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes:
        center, scale = _box2cs(cfg, bbox)

        # prepare data
        data = {
            'img_or_path':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)

    return result['preds'], result['output_heatmap']


def inference_top_down_pose_model(model,
                                  img_or_path,
                                  person_results,
                                  bbox_thr=None,
                                  format='xyxy',
                                  dataset='TopDownCocoDataset',
                                  return_heatmap=False,
                                  outputs=None):
    """Inference a single image with a list of person bounding boxes.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr: Threshold for bounding boxes. Only bboxes with higher scores
            will be fed into the pose detector. If bbox_thr is None, ignore it.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None

    Returns:
        list[dict]: The bbox & pose info,
            Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
        list[dict[np.ndarray[N, K, H, W] | torch.tensor[N, K, H, W]]]:
            Output feature maps from layers specified in `outputs`.
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = _xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = _xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_pose_model(
            model,
            img_or_path,
            bboxes_xywh,
            dataset,
            return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results
    # return pose_results, returned_outputs


def inference_bottom_up_pose_model(model,
                                   img_or_path,
                                   pose_nms_thr=0.9,
                                   return_heatmap=False,
                                   outputs=None):
    """Inference a single image.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        pose_nms_thr (float): retain oks overlap < pose_nms_thr, default: 0.9.
        return_heatmap (bool) : Flag to return heatmap, default: False.
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None.

    Returns:
        list[ndarray]: The predicted pose info.
            The length of the list is the number of people (P).
            Each item in the list is a ndarray, containing each person's
            pose (ndarray[Kx3]): x, y, score.
        list[dict[np.ndarray[N, K, H, W] | torch.tensor[N, K, H, W]]]:
            Output feature maps from layers specified in `outputs`.
            Includes 'heatmap' if `return_heatmap` is True.
    """
    pose_results = []
    returned_outputs = []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = {
        'img_or_path': img_or_path,
        'dataset': 'coco',
        'ann_info': {
            'image_size':
            cfg.data_cfg['image_size'],
            'num_joints':
            cfg.data_cfg['num_joints'],
            'flip_index':
            [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        }
    }

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'].data[0]

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # forward the model
        with torch.no_grad():
            result = model(
                img=data['img'],
                img_metas=data['img_metas'],
                return_loss=False,
                return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = result['output_heatmap']

        returned_outputs.append(h.layer_outputs)

        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            pose_results.append({
                'keypoints': pred[:, :3],
                'score': result['scores'][idx],
                'area': area,
            })

        # pose nms
        keep = oks_nms(pose_results, pose_nms_thr, sigmas=None)
        pose_results = [pose_results[_keep] for _keep in keep]

    return pose_results, returned_outputs


def vis_pose_result(model,
                    img,
                    result,
                    radius=5,
                    thickness=3,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset in 'AnimalPoseDataset':
        skeleton = [[1, 2], [2, 3], [3, 10], [10, 17], [3, 4], [3, 7], [10, 11], [10, 14],
                    [4, 5], [5, 6], [7, 8], [8, 9], [11, 12], [12, 13], [14, 15], [15, 16]]

        pose_kpt_color = palette[[17] * 17]
        pose_limb_color = palette[[18] * 16]

    else:
        raise NotImplementedError()

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img
