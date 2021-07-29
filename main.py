import os
import numpy as np
import time
from keypoint_detector.mmpose.apis.inference import inference_top_down_pose_model, init_pose_model, vis_pose_result
from inference_utils.evaluation_pck import get_single_pck, get_pck_json
from PIL import Image


start = time.time()

img_root = 'data/challenge_test_images/'
ann_root = 'data/challenge_test_annotations/'
dataset = 'AnimalPoseDataset'

keypoint_detector_config = 'keypoint_detector/configs/hrnet_w48_animalpose_256x256.py'
keypoint_detector_model = 'weights/epoch_210.pth'

preds = []
bboxes = []

eval_pck = True
visualization_result = False

entries = os.listdir(img_root)

for entry in entries:
    img = Image.open(img_root + entry)
    bboxes.append([[0, 0, img.size[0], img.size[1]]])

model = init_pose_model(keypoint_detector_config, keypoint_detector_model)

for i, bbox in enumerate(bboxes):
    img_path = img_root + entries[i]
    bbox = np.array(bbox)

    result = inference_top_down_pose_model(model, img_path, bbox, dataset=dataset)

    if visualization_result:
        out_file_name = 'results/' + img_path.split('/')[-1]
        vis_pose_result(model, img_path, result, dataset=dataset, out_file=out_file_name)

    if eval_pck:
        preds.append(result[0][:, :-1])

if eval_pck:
    pck = get_pck_json(preds, ann_root)
    print(pck)

print('time : ', time.time() - start)
