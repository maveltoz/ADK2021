import os
import numpy as np
import time
import xml.etree.ElementTree as ET
from mmpose.core import keypoint_pck_accuracy
from object_detector import detect
from keypoint_detector.mmpose.apis.inference import inference_top_down_pose_model, init_pose_model, vis_pose_result


start = time.time()

img_root = 'data/test/image/'
ann_root = 'data/test/annotation/'
dataset = 'AnimalPoseDataset'
images = []
preds = []
gts = []
masks = []
thr = 0.2
normalize = []

entries = os.listdir(img_root)

for entry in entries:
    image = img_root + entry
    images.append(image)

bboxes = detect.main(img_root)

model = init_pose_model('keypoint_detector/configs/hrnet_w48_animalpose_256x256.py', 'weights/keypoint_detector/best.pth')
# model = init_pose_model('keypoint_detector/configs/hrnet_w48_animalpose_256x256.py', 'work_dirs/hrnet_w48_animalpose_256x256/best.pth')
# model = init_pose_model('keypoint_detector/configs/hrnet_w48_animalpose_256x256.py', 'weights/keypoint_detector/hrnet_w48_animalpose_256x256-34644726_20210426.pth')


for i, bbox in enumerate(bboxes):
    person_results = []
    for box in bbox:
        person_results.append({'bbox': box})

    img_path = images[i]

    person_results = np.array(person_results)
    result = inference_top_down_pose_model(model, img_path, person_results, dataset=dataset)

    area_max = 0.0
    keypoints = result[0]['keypoints']

    for res in result:
        x1 = res['bbox'][0]
        y1 = res['bbox'][1]
        x2 = res['bbox'][2]
        y2 = res['bbox'][3]
        area = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        if area >= area_max:
            area_max = area
            keypoints = res['keypoints']

    out_file_name = 'results/' + img_path.split('/')[-1]
    vis_pose_result(model, img_path, result, dataset=dataset, out_file=out_file_name)

    preds.append(keypoints[:, :-1])

preds = np.array(preds)

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
print(pck)

print('time : ', time.time() - start)
