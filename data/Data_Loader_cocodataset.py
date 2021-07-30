# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import json
from augmentations import *

import re
import collections

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
           json_file (string): json_file 파일의 경로
           root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
           transform (callable, optional): 샘플에 적용될 Optional transform
        """
        super().__init__()

        with open(json_file, 'r') as keypoint:
            keypoint_data = json.load(keypoint)
        self.landmarks_frame = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in keypoint_data.items()]))
        self.root_dir = root_dir
        self.transform = transform
        self.anno = self.landmarks_frame['annotations'].dropna()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame['images'][idx]['file_name'])

        if os.path.isfile(img_name):
            image = io.imread(img_name)

            land_num = []
            for i in range(len(self.anno)):
                if self.landmarks_frame['images'][idx]['id'] == self.anno[i]['image_id']:
                    land_num.append(i)

            landmarks = []
            for i, landmark in enumerate(land_num):
                landmarks.append(self.anno[landmark]['keypoints'])

            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 17, 3)
            
            sample = {'image': image, 'landmarks': landmarks}

            if self.transform:
                sample = self.transform(sample)

        return sample

annot_dir = 'D:\/\Downloads\/coco dataset\/annotations\/person_keypoints_val2017.json'
image_dir = 'D:\/\Downloads\/coco dataset\/val2017\/'
# transformed_dataset = FaceLandmarksDataset(json_file=annot_dir, root_dir=image_dir, transform=transforms.Compose([Rescale(512), RandomCrop(448), ToTensor()]))
transformed_dataset = FaceLandmarksDataset(json_file=annot_dir, root_dir=image_dir, transform=transforms.Compose([Perspective(), ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None)

# np_str_obj_array_pattern = re.compile(r'[SaUO]')
# default_collate_err_msg_format = (
    # "default_collate: batch must contain tensors, numpy arrays, numbers, "
    # "dicts or lists; found {}")

# def collate_fn(batch):
    # elem = batch[0]  # {'image', 'landmarks'}, image channal, landmarks size
    # elem_type = type(elem)
    # mybatch_1 = batch[1]
    # mybatch_2 = batch[2]
    # # print(mybatch.batch_size, mybatch)
    # print(mybatch['landmarks'])
    # print(mybatch_1['landmarks'])
    # print(mybatch_2['landmarks'])
    # print('mybatch_len', len(mybatch))
    # print('mybatch_1_len', len(mybatch_1))
    # print('mybatch_2_len', len(mybatch_2))
    # print('mybatch_type', mybatch_type)

    #print('data', data)
    # transposed = zip(*batch)
    # print(transposed)
    # for samples in transposed:
    #     print(samples[0])
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             numel = sum([x.numel() for x in batch])
#             print('numel', numel)
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#             print('out', out)
#         return torch.stack(batch, 0, out=out)                               ######
#     elif isinstance(elem, collections.abc.Mapping):
#         return {key: collate_fn([d[key] for d in batch]) for key in elem}   ######
# 
# #collate_fn(dataloader)
# dataloader2 = DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

# for data in dataloader2:
#     print(len(data))

# for i_batch, sample_batched in enumerate(dataloader2):
#     # landmarks_batch = sample_batched['landmarks']
#     print(i_batch)
#     print(sample_batched)

#for data in dataloader:
#    print(data['landmarks'].shape)


# def make_batch(samples):
#     inputs = [sample['image'] for sample in samples]
#     labels = [sample['landmarks'] for sample in samples]
#     padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
#     return {'image': padded_inputs.contiguous(), 'landmarks': torch.stack(labels).contiguous()}

# 배치하는 과정을 보여주는 함수입니다.
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    print(landmarks_batch.shape)
    batch_size = len(images_batch)
    land_batch_size = len(landmarks_batch)
    im_size = images_batch.size(3)

    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
       
        plt.scatter(landmarks_batch[i, :, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, :, 1].numpy() + grid_border_size, s=10, marker='.', c='r')
        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):

    landmarks_batch = sample_batched['landmarks']

    if i_batch == 0:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
