import json, os
from collections import OrderedDict
from tqdm import tqdm
import argparse

image_dir = "./data/animalpose/images/"
json_dir = "./data/animalpose/annotations/"
result_dir = "./data/animalpose/annotations_/"

file_data = OrderedDict()
jsonfile = os.listdir(json_dir)

# data split
parser = argparse.ArgumentParser()
parser.add_argument('--train_num', default=6000, type=int, dest='train_num')
parser.add_argument('--valid_num', default=2000, type=int, dest='validation_num')
parser.add_argument('--test_num', default=len(jsonfile) - 1, type=int, dest='test_num')
args = parser.parse_args()

images = []
annotations = []
categories = []

def split_data(data):
    global images, annotations, categories
    global file_data

    file_data = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(result_dir + 'animalpose_' + data + '.json', 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')

    images = []
    annotations = []
    categories = []

for i, data_name in enumerate(tqdm(jsonfile)):
    with open(json_dir + data_name, 'r') as json_file:
        cow_data = json.load(json_file)
    id = int(cow_data['label_info']['image']['file_name'].split('livestock_cow_keypoints_')[1].split('.jpg')[0])

    images.append({'file_name': cow_data['label_info']['image']['file_name'],
                               'height': cow_data['label_info']['image']['height'],
                               'width': cow_data['label_info']['image']['width'],
                               'id': id,
                               })
    annotations.append({'image_id': id, 'id': (id + 100000), 'category_id': cow_data['label_info']['annotations'][0]['category_id'],
                                     'keypoints': cow_data['label_info']['annotations'][0]['keypoints'],
                                     'bbox': cow_data['label_info']['annotations'][0]['bbox']})
    categories = [cow_data['label_info']['categories'][0]]

    if i == (args.train_num - 1):
        split_data('train')

    elif i == (args.train_num + args.validation_num):
        split_data('val')

    elif i == args.test_num:
        split_data('test')
