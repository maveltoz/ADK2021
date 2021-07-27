import json, os
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

image_dir = 'data/animalpose/images/'
json_dir = 'data/animalpose/annotations/'
result_dir = 'data/animalpose/result/'

imagefile = os.listdir(image_dir)
jsonfile = os.listdir(json_dir)
# ######################################################################################################################

# https://github.com/hmallen/numpyencoder
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    numpy_encoder = NumpyEncoder()

transform = A.Compose([
    # A.RandomBrightnessContrast(p=0.2),
    A.Affine()
    # A.SafeRotate()
    # A.HorizontalFlip()
    # A.ShiftScaleRotate()
], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False, angle_in_degrees=True))

for data_name in tqdm(jsonfile):
    with open(json_dir + data_name, 'r') as json_file:
        cow_data = json.load(json_file)
        file_data = cow_data

        for i in tqdm(range(len(cow_data['images']))):
            class_labels = cow_data['categories'][0]['keypoints_name']
            image_name = cow_data['images'][i]['file_name']
            keypoints = np.array(cow_data['annotations'][i]['keypoints'])

            image = cv2.imread(image_dir + image_name)
            keypoints = keypoints.reshape(-1, 3)

            # ori_image = image
            #
            # for key in keypoints:
            #     x = key[0]
            #     y = key[1]
            #     ori_image = cv2.circle(ori_image, (x, y), 5, (0, 0, 255), -1)
            # cv2.imwrite(result_dir + 'ori/' + image_name, ori_image) # livestock_cow_keypoints_000001

            transformed = transform(image=image, keypoints=keypoints, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            transformed_class_labels = transformed['class_labels']

            # for key in transformed_keypoints:
            #     x = int(key[0])
            #     y = int(key[1])
            #     transformed_image = cv2.circle(transformed_image, (x, y), 5, (0, 0, 255), -1)
            #
            # cv2.imwrite(result_dir + 'aug/' + image_name, transformed_image)
            cv2.imwrite('data/animalpose/Affine/' + image_name, transformed_image) # livestock_cow_keypoints_000001

            new_transformed_keypoints = np.array(transformed_keypoints)
            new_transformed_keypoints = new_transformed_keypoints.reshape(-1)

            file_data['annotations'][i]['keypoints'] = list(new_transformed_keypoints)

    with open('data/animalpose/Affine/' + data_name, 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t', cls=NumpyEncoder)
# ######################################################################################################################
