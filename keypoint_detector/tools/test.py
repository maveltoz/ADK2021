import argparse
import os
import warnings

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', help='output result file', default=True)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def get_pck_json(preds, ann_root='data/test/challenge_annotations/'):
    preds = np.array(preds)
    gts = []
    masks = []
    thr = 0.2
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False,
        drop_last=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)

    wrap_fp16_model(model)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    if args.eval:
        preds = []

        for output in outputs:
            for keypoints in output['preds']:
                preds.append(np.array(keypoints)[:, :-1])

        preds = np.array(preds)
        ann_root = 'data/challenge_test_annotations/'

        result = get_pck_json(preds, ann_root)
        print('\n=> evaluation pck result:')
        print(result)

    if args.out:
        out_file_path = "./result"
        if not os.path.isdir(out_file_path):
            os.mkdir(out_file_path)

        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, out_file_path + '/' + args.out)


if __name__ == '__main__':
    main()
