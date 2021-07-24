import os
import numpy as np
import time
from object_detector import detect
from keypoint_detector.mmpose.apis.inference import inference_top_down_pose_model, init_pose_model, vis_pose_result
from inference_utils.evaluation_pck import get_single_pck, get_pck, get_pck_json
import cv2


start = time.time()

img_root = 'data/animalpose/images/'
ann_root = 'data/test/challenge_annotations/'
dataset = 'AnimalPoseDataset'

object_detect_model = 'weights/object_detector/yolov5x6.pt'
keypoint_detector_config = 'keypoint_detector/configs/hrnet_w48_animalpose_256x256.py'
keypoint_detector_model = 'work_dirs/hrnet_w48_animalpose_256x256/epoch_210.pth'

images = []
preds = []
imgs = []

eval_pck = True
visualization_result = False

entries = os.listdir(img_root)

for entry in entries:
    image = img_root + entry
    images.append(image)
    img = cv2.imread(image)
    imgs.append([[0, 0, img.shape[1], img.shape[0]]])

# bboxes = detect.main(img_root, object_detect_model)
bboxes = imgs
# print(bboxes)

model = init_pose_model(keypoint_detector_config, keypoint_detector_model)

for i, bbox in enumerate(bboxes):
    person_results = []
    for box in bbox:
        person_results.append({'bbox': box})

    img_path = images[i]

    person_results = np.array(person_results)
    result = inference_top_down_pose_model(model, img_path, person_results, dataset=dataset)

    out_file_name = 'results/' + img_path.split('/')[-1]
    if visualization_result:
        vis_pose_result(model, img_path, result, dataset=dataset, out_file=out_file_name)

    if eval_pck:
        single_pck = get_single_pck(result)
        preds.append(single_pck)

if eval_pck:
    pck = get_pck_json(preds, ann_root)
    print(pck)

print('time : ', time.time() - start)
