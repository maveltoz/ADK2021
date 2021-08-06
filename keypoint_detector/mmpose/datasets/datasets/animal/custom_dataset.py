import os
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from .animal_base_dataset import AnimalBaseDataset
from PIL import Image
from mmpose.core import keypoint_pck_accuracy


@DATASETS.register_module()
class CustomDataset(AnimalBaseDataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_root,
                 data_cfg,
                 pipeline,
                 test_mode=True):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.img_root = img_root
        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            warnings.warn(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['flip_pairs'] = [[3, 6], [4, 7], [5, 8], [10, 13], [11, 14], [12, 15]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 7, 10, 11, 14)
        self.ann_info['lower_body_ids'] = (5, 6, 8, 9, 12, 13, 15, 16)

        self.ann_info['use_different_joint_weights'] = True
        self.ann_info['joint_weights'] = np.array(
            [
                1., 1., 1., 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1., 1.2, 1.5, 1., 1.2, 1.5, 1
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.sigmas = np.array([
            .35, 1.0, 1.0, 1.07, .87, .89, 1.07, .87, .89, 1.0, 1.07, .87, .89, 1.07, .87, .89, 1.0
        ]) / 10.0

        self.bbox = []

        self.dataset_name = 'animalpose'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []

        entries = os.listdir(self.img_root)
        entries.sort()

        for img_path in entries:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_path))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_path):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img = Image.open(self.img_root + img_path)
        width = img.size[0]
        height = img.size[1]

        x, y, w, h = (0, 0, width, height)

        self.bbox.append([w, h])

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width - 1, x1 + max(0, w - 1))
        y2 = min(height - 1, y1 + max(0, h - 1))

        bbox = [x1, y1, x2 - x1, y2 - y1]

        center, scale = self._xywh2cs(*bbox)

        image_file = os.path.join(self.img_root, img_path)

        rec = []

        rec.append({
            'image_file': image_file,
            'center': center,
            'scale': scale,
            'bbox': bbox,
            'rotation': 0,
            'dataset': self.dataset_name,
            'bbox_score': 1,
            'bbox_id': 0
        })

        return rec

    def evaluate(self, outputs, ann_root='data/challenge_test_annotations/'):
        preds = []

        for output in outputs:
            for keypoints in output['preds']:
                preds.append(np.array(keypoints)[:, :-1])

        preds = np.array(preds)

        result = self.get_pck_json(preds, ann_root)

        return result

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

    def get_pck_json(self, preds, ann_root='data/test/challenge_annotations/'):
        preds = np.array(preds)
        gts = []
        masks = []
        thr = 0.35
        normalize = []

        entries = os.listdir(ann_root)
        entries.sort()

        for entry in entries:
            with open(ann_root + entry) as annotation_file:
                annotation = json.load(annotation_file)['label_info']

                w = int(annotation['image']['width'])
                h = int(annotation['image']['height'])
                bbox_thr = np.max([w, h])
                normalize.append([bbox_thr, bbox_thr])
                
                # kpt1_x = int(annotation['annotations'][0]['keypoints'][6])
                # kpt1_y = int(annotation['annotations'][0]['keypoints'][7])
                # kpt2_x = int(annotation['annotations'][0]['keypoints'][27])
                # kpt2_y = int(annotation['annotations'][0]['keypoints'][28])
                # bbox_thr_1 = np.max([kpt1_x, kpt1_y])
                # bbox_thr_2 = np.max([kpt2_x, kpt2_y])
                # normalize.append([bbox_thr_1, bbox_thr_2])

                now_gt = []
                now_mask = []

                for k in range(17):
                    x = annotation['annotations'][0]['keypoints'][3 * k]
                    y = annotation['annotations'][0]['keypoints'][3 * k + 1]
                    z = annotation['annotations'][0]['keypoints'][3 * k + 2]
                    gt = [x, y]
                    now_gt.append(gt)
                    if z == 2:
                        mask = True
                    else:
                        mask = True
                        # mask = False
                    now_mask.append(mask)
                gts.append(now_gt)
                masks.append(now_mask)

        gts = np.array(gts)
        masks = np.array(masks)
        normalize = np.array(normalize)

        pck = keypoint_pck_accuracy(preds, gts, masks, thr, normalize)

        return pck

    def get_bbox(self):
        return self.bbox
