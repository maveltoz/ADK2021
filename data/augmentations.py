import cv2
import torch
from skimage import io, transform
import numpy as np

import albumentations as A
import math

class Rescale(object):
    """주어진 사이즈로 샘플크기를 조정합니다.
    Args:
        output_size(tuple or int) : 원하는 사이즈 값
            tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
            int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.
        Args:
            output_size (tuple or int): 줄이고자 하는 크기입니다.
                            int라면, 정사각형으로 나올 것 입니다.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,))
        left = torch.randint(0, w - new_w, (1,))

        image = image[top: top + new_h, left: left + new_w] # 이게 뭐지?
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class Rotation(object):
    """샘플데이터를 0 ~ 360도 사이로 랜덤하게 회전
            Args:
                output_size (tuple or int):
        """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]

        (cX, cY) = (w // 2, h // 2)  # center point

        M = cv2.getRotationMatrix2D((cX, cY), angle=-self.output_size, scale=1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        image = cv2.warpAffine(image, M, (nW, nH))

        x = landmarks[:, :, 0]
        y = landmarks[:, :, 1]

        new_X = x * cos - y * sin + M[0, 2]
        new_Y = x * sin + y * cos + M[1, 2]
        landmarks[:, :, 0] = new_X
        landmarks[:, :, 1] = new_Y

        return {'image': image, 'landmarks': landmarks}

class Filping(object):
    """샘플데이터를 상하/좌우 반전
            Args:
                output_size (tuple or int):
        """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]

        print(landmarks)
        if self.output_size == 0:   # 상하 반전
            image = cv2.flip(image, self.output_size)
            landmarks[:, :, 1] = np.abs(landmarks[:, :, 1] - h)
        elif self.output_size == 1: # 좌우 반전
            image = cv2.flip(image, self.output_size)
            landmarks[:, :, 0] = np.abs(landmarks[:, :, 0] - w)


        return {'image': image, 'landmarks': landmarks}


class Brightness(object):
    """샘플데이터의 밝기를 조절
            Args:
                output_size (tuple or int):
        """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        image = cv2.add(image, (self.output_size, self.output_size, self.output_size, 0))

        return {'image': image, 'landmarks': landmarks}

class Shifting(object):
    """샘플데이터를 10px씩 상하좌우로 움직여준다
            Args:
                output_size (tuple or int):
        """
    def __init__(self, w, h):
        assert isinstance((w, h), (int, tuple))
        self.w = w
        self.h = h

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]
        M = np.float32([[1, 0, self.w], [0, 1, self.h]])

        image = cv2.warpAffine(image, M, (w, h))

        landmarks[:, :, 0] = landmarks[:, :, 0] + self.w
        landmarks[:, :, 1] = landmarks[:, :, 1] + self.h

        return {'image': image, 'landmarks': landmarks}

class Perspective(object):
    """샘플데이터를 -20 ~ 20만큼 강제로 찌그러뜨린다
            Args:
                output_size (tuple or int):
        """
    def __init__(self):
        print()
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]
        random_num = np.random.randint(-20, 20)
        random_x = np.random.randint(-20, 20)
        random_y = np.random.randint(-20, 20)
        random_z = np.random.randint(-20, 20)
        print(random_num, random_x, random_y, random_z)

        # [x,y] 좌표점을 4x2의 행렬로 작성
        # 좌표점은 좌상->좌하->우상->우하
        src = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        dst = np.float32([[0 + random_x, 0 + random_y], [0 + random_z, h + (random_x + random_y)],
                          [w + (random_x + random_z), 0 + (random_y + random_z)], [w + (random_x - random_y), h + (random_x - random_z)]])
        # dst = np.float32([[random_num, 0], [0, h],
        #                   [w + random_num, 0], [w, h]])
        M = cv2.getPerspectiveTransform(src, dst)
        perspective_image = cv2.warpPerspective(image, M, (0, 0))

        x = landmarks[:, :, 0]
        y = landmarks[:, :, 1]

        print(M)

        # 공식 >> https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=samsjang&logNo=220504966397&view=img_13
        newX = (M[0][0] * x + M[0][1] * y + M[0][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])
        newY = (M[1][0] * x + M[1][1] * y + M[1][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])

        landmarks[:, :, 0] = newX
        landmarks[:, :, 1] = newY

        return {'image': perspective_image, 'landmarks': landmarks}

# class Perspective(object):
#     """샘플데이터를 -20 ~ 20만큼 강제로 찌그러뜨린다
#             Args:
#                 output_size (tuple or int):
#         """
#     def __init__(self):
#         print()
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
#         (h, w) = image.shape[:2]
#         random_num = np.random.randint(-20, 20)
#         random_x = np.random.randint(-20, 20)
#         random_y = np.random.randint(-20, 20)
#         random_z = np.random.randint(-20, 20)
#         print(random_num, random_x, random_y, random_z)
#
#         # [x,y] 좌표점을 4x2의 행렬로 작성
#         # 좌표점은 좌상->좌하->우상->우하
#         src = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
#         # dst = np.float32([[0 + random_x, 0 + random_y], [0 + random_z, h + (random_x + random_y)],
#         #                   [w + (random_x + random_z), 0 + (random_y + random_z)], [w + (random_x - random_y), h + (random_x - random_z)]])
#         dst = np.float32([[random_num, 0], [0, h],
#                           [w + random_num, 0], [w, h]])
#         M = cv2.getPerspectiveTransform(src, dst)
#         perspective_image = cv2.warpPerspective(image, M, (0, 0))
#
#         x = landmarks[:, :, 0]
#         y = landmarks[:, :, 1]
#         hypotenuse = math.sqrt(pow(random_num, 2) + pow(h, 2))
#         cos = h / hypotenuse
#         tanA = random_num / h
#         a = (h - y) * math.tan(tanA)
#         # print(a)
#         new_X = x + a
#         new_Y = y
#
#         landmarks[:, :, 0] = new_X
#         landmarks[:, :, 1] = new_Y
#
#         # print(perspective_image)
#         # print(perspective_image.shape)
#         #
#         # landmarks_reshape = landmarks.reshape(-1, 2)
#         # print(landmarks_reshape)
#         # print(landmarks_reshape.shape)
#         # landmarks_newaxis = landmarks_reshape[:, :, np.newaxis]
#         # print(landmarks_newaxis)
#         # print(landmarks_newaxis.shape)
#         # new_landmarks = cv2.warpPerspective(landmarks_newaxis, M, (0, 0))
#         # print(new_landmarks.shape)
#
#
#         # test_array = np.array([[[1, 2], [5, 8]], [[7, 3], [6, 9]]])
#         # print(test_array.shape)
#         # test_reshape = test_array.reshape(-1, 2)
#         # print(test_reshape)
#         # print(test_reshape.shape)
#         return {'image': perspective_image, 'landmarks': landmarks}

class Stretching(object):
    """샘플데이터를 강제로 1 ~ 1.3배로 늘린다. 센터 기준
                Args:
                    output_size (tuple or int):
            """
    def __init__(self, output_size):
        assert isinstance(output_size, (float, int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]

        (cX, cY) = (w // 2, h // 2)  # center point
        print(cX, cY)
        M = cv2.getRotationMatrix2D((cX, cY), angle=0.0, scale=self.output_size)

        image = cv2.warpAffine(image, M, (0, 0))

        landmarks[:, :, 0] = (landmarks[:, :, 0] - cX) * self.output_size + cX
        landmarks[:, :, 1] = (landmarks[:, :, 1] - cY) * self.output_size + cY

        return {'image': image, 'landmarks': landmarks}

class RGB_Average(object):
    """샘플데이터의 RGB값의 평균을 뺀다
                   Args:
                        output_size (tuple or int):
                """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]
        Red = image[:, :, 0]
        Green = image[:, :, 1]
        Blue = image[:, :, 2]
        red = []
        green = []
        blue = []

        for i in range(h):
            red.append(sum(Red[i]))
            green.append(sum(Green[i]))
            blue.append(sum(Blue[i]))
        R_aver = sum(red) / (h * w)
        G_aver = sum(green) / (h * w)
        B_aver = sum(blue) / (h * w)

        get_R_aver = image[:, :, 0] < R_aver
        get_G_aver = image[:, :, 1] < G_aver
        get_B_aver = image[:, :, 2] < B_aver
        image[:, :, 0] = image[:, :, 0 ] * get_R_aver
        image[:, :, 1] = image[:, :, 1] * get_G_aver
        image[:, :, 2] = image[:, :, 2] * get_B_aver

        print(image[:, :, 0 ] < R_aver)
        print('image[0]', image[0])

        return {'image': image, 'landmarks': landmarks}

class Blur(object):
    """샘플데이터의 Blur filter
                   Args:
                        output_size (tuple or int):
                """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        image = cv2.blur(image, (self.output_size, self.output_size))

        return {'image': image, 'landmarks': landmarks}

class GaussianBlur(object):
    """샘플데이터의 GaussianBlur filter
                   Args:
                        output_size (tuple or int):
                """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        image = cv2.GaussianBlur(image, (self.output_size, self.output_size), 0)    # 0 >> ((너비-1)0.5-1)0.3+0.8

        return {'image': image, 'landmarks': landmarks}

class Sharp(object):
    """샘플데이터 선명하게
                   Args:
                        output_size (tuple or int):
                """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        # 커널 생성(대상이 있는 픽셀을 강조)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        image = cv2.filter2D(image, -1, kernel)

        return {'image': image, 'landmarks': landmarks}

class Histogram_Equalization(object):
    """샘플데이터 대비 높이기
                   Args:
                        output_size (tuple or int):
                """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]

        # YUV 컬로 포맷으로 변환
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # 히스토그램 평활화 적용
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        # #RGB로 변환
        image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

        return {'image': image_rgb, 'landmarks': landmarks}

class grabCut(object):
    """샘플데이터 배경 제거
                   Args:
                        output_size (tuple or int):
                """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]

        # 사각형 좌표: 시작점의 x, y, 높이, 너비
        rectangle = (0, 0, h, w)

        # 초기 마스크 생성
        mask = np.zeros(image.shape[:2], np.uint8)

        # grabCut에 사용할 임시 배열 생성
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # grabCut 실행
        cv2.grabCut(image,  # 원본 이미지
                    mask,  # 마스크
                    rectangle,  # 사각형
                    bgdModel,  # 배경을 위한 임시 배열
                    fgdModel,  # 전경을 위한 임시 배열
                    100,  # 반복 횟수
                    cv2.GC_INIT_WITH_RECT)  # 사각형을 위한 초기화
        # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
        mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 이미지에 새로운 마스크를 곱행 배경을 제외
        image_rgb_nobg = image * mask_2[:, :, np.newaxis]

        return {'image': image_rgb_nobg, 'landmarks': landmarks}

class Canny(object):
    """샘플데이터 경계선 감지
                   Args:
                        output_size (tuple or int):
                """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        (h, w) = image.shape[:2]

        # 픽셀 강도의 중간값을 계산
        median_intensity = np.median(image)

        # 중간 픽셀 강도에서 위아래 1 표준편차 떨어진 값을 임곗값으로 지정
        lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
        upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

        # Canny edge detection 적용
        image_canny = cv2.Canny(image, lower_threshold, upper_threshold)
        image_canny = image_canny[:, :, np.newaxis]
        print(image_canny.shape)
        return {'image': image_canny, 'landmarks': landmarks}

class Albumentations(object):
    # pip install -U albumentations
    # pip install -U --user albumentations
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        transform = A.Compose([A.RandomCrop(width=256, height=256),
                               A.HorizontalFlip(p=0.5),
                               A.RandomBrightnessContrast(p=0.2),
                               ], keypoint_params=A.KeypointParams(format='xy'))
        transformed = transform(image=image, keypoints=landmarks[0])
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        print(transformed_keypoints)

        return {'image': transformed_image, 'landmarks': landmarks}


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'][:, :, :2]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(image), 'landmarks':torch.from_numpy(landmarks)}
