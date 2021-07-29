import os
import numpy as np
import json
import xml.etree.ElementTree as ET
from mmpose.core import keypoint_pck_accuracy


def get_single_pck(result):
    area_max = 0.0
    keypoints = result[0]['keypoints']

    for res in result:
        x1 = res['bbox'][0]
        y1 = res['bbox'][1]
        x2 = res['bbox'][2]
        y2 = res['bbox'][3]
        area = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if area >= area_max:
            area_max = area
            keypoints = res['keypoints']

    return keypoints[:, :-1]


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


def get_pck(preds, ann_root='data/test/annotation/'):
    preds = np.array(preds)
    gts = []
    masks = []
    thr = 0.2
    normalize = []

    entries = os.listdir(ann_root)

    for entry in entries:
        doc = ET.parse(ann_root + entry)
        root = doc.getroot()

        for b in root.iter('visible_bounds'):
            w = int(float(b.attrib['width']))
            h = int(float(b.attrib['height']))
            bbox_thr = np.max([w, h])
            normalize.append([bbox_thr, bbox_thr])

        now_gt = []
        now_mask = []

        for k in root.iter('keypoint'):
            x = int(float(k.attrib['x']))
            y = int(float(k.attrib['y']))
            z = int(float(k.attrib['visible']))
            gt = [x, y]
            now_gt.append(gt)
            if z == 1:
                mask = True
            else:
                mask = False
            now_mask.append(mask)
        gts.append(now_gt)
        masks.append(now_mask)

    gts = np.array(gts)
    masks = np.array(masks)
    normalize = np.array(normalize)

    pck = keypoint_pck_accuracy(preds, gts, masks, thr, normalize)

    return pck
