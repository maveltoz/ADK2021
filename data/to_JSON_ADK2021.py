import json, os
from collections import OrderedDict

image_dir = 'C:/Users/minu/Documents/PycharmProjects/ADK2021/data/Training_Data/images/'
json_dir = 'C:/Users/minu/Documents/PycharmProjects/ADK2021/data/Training_Data/annotations/'
result_dir = 'data/animalpose/annotations/'
# image_dir = 'C:/Users/minu/Documents/PycharmProjects/ADK2021/data/Training_Data/test_img/'
# json_dir = 'C:/Users/minu/Documents/PycharmProjects/ADK2021/data/Training_Data/test_anno/'

file_data = OrderedDict()
jsonfile = os.listdir(json_dir)

# data split
train_num = 6000
validation_num = 2000
test_num = len(jsonfile) - 1

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

for i, data_name in enumerate(jsonfile):
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

    if i == (train_num - 1):
        split_data('train')

    elif i == (train_num + validation_num):
        split_data('val')

    elif i == test_num:
        split_data('test')
