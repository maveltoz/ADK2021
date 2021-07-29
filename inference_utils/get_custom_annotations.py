import json


def get_custom_json(annotation_file, out_file):
    dataset = json.load(open(annotation_file, 'r'))

    for data in dataset['annotations']:
        keypoints = data['keypoints']
        new_keypoints = []
        num_keypoints = 0

        for i in range(17):
            if i == 2:
                new_keypoints.append(keypoints[7 * 3])
                new_keypoints.append(keypoints[7 * 3 + 1])
                new_keypoints.append(keypoints[7 * 3 + 2])
            elif i == 3:
                new_keypoints.append(keypoints[9 * 3])
                new_keypoints.append(keypoints[9 * 3 + 1])
                new_keypoints.append(keypoints[9 * 3 + 2])
            elif i == 4:
                new_keypoints.append(keypoints[13 * 3])
                new_keypoints.append(keypoints[13 * 3 + 1])
                new_keypoints.append(keypoints[13 * 3 + 2])
            elif i == 5:
                new_keypoints.append(keypoints[17 * 3])
                new_keypoints.append(keypoints[17 * 3 + 1])
                new_keypoints.append(keypoints[17 * 3 + 2])
            elif i == 6:
                new_keypoints.append(keypoints[8 * 3])
                new_keypoints.append(keypoints[8 * 3 + 1])
                new_keypoints.append(keypoints[8 * 3 + 2])
            elif i == 7:
                new_keypoints.append(keypoints[12 * 3])
                new_keypoints.append(keypoints[12 * 3 + 1])
                new_keypoints.append(keypoints[12 * 3 + 2])
            elif i == 8:
                new_keypoints.append(keypoints[16 * 3])
                new_keypoints.append(keypoints[16 * 3 + 1])
                new_keypoints.append(keypoints[16 * 3 + 2])
            elif i == 10:
                new_keypoints.append(keypoints[11 * 3])
                new_keypoints.append(keypoints[11 * 3 + 1])
                new_keypoints.append(keypoints[11 * 3 + 2])
            elif i == 11:
                new_keypoints.append(keypoints[15 * 3])
                new_keypoints.append(keypoints[15 * 3 + 1])
                new_keypoints.append(keypoints[15 * 3 + 2])
            elif i == 12:
                new_keypoints.append(keypoints[19 * 3])
                new_keypoints.append(keypoints[19 * 3 + 1])
                new_keypoints.append(keypoints[19 * 3 + 2])
            elif i == 13:
                new_keypoints.append(keypoints[10 * 3])
                new_keypoints.append(keypoints[10 * 3 + 1])
                new_keypoints.append(keypoints[10 * 3 + 2])
            elif i == 14:
                new_keypoints.append(keypoints[14 * 3])
                new_keypoints.append(keypoints[14 * 3 + 1])
                new_keypoints.append(keypoints[14 * 3 + 2])
            elif i == 15:
                new_keypoints.append(keypoints[18 * 3])
                new_keypoints.append(keypoints[18 * 3 + 1])
                new_keypoints.append(keypoints[18 * 3 + 2])
            elif i == 16:
                new_keypoints.append(keypoints[6 * 3])
                new_keypoints.append(keypoints[6 * 3 + 1])
                new_keypoints.append(keypoints[6 * 3 + 2])
            else:
                new_keypoints.append(0.0)
                new_keypoints.append(0.0)
                new_keypoints.append(0.0)

        for j in range(17):
            if new_keypoints[j * 3] > 0:
                num_keypoints += 1

        data['keypoints'] = new_keypoints
        data['num_keypoints'] = num_keypoints

    dataset['categories'][0]['keypoints'] = ['head', 'neck', 'withers', 'R_F_Elbow', 'R_F_Knee', 'R_F_Paw', 'L_F_Elbow',
                                             'L_F_Knee', 'L_F_Paw', 'Back_Waist', 'R_B_Elbow', 'R_B_Knee', 'R_B_Paw',
                                             'L_B_Elbow', 'L_B_Knee', 'L_B_Paw', 'TailBase']

    skeleton = [[1, 2], [2, 3], [3, 10], [10, 17], [3, 4], [3, 7], [10, 11], [10, 14], [4, 5], [5, 6], [7, 8],
                [8, 9], [11, 12], [12, 13], [14, 15], [15, 16]]

    dataset['categories'][0]['skeleton'] = skeleton

    with open(out_file, 'w', encoding='utf-8') as make_file:
        json.dump(dataset, make_file, ensure_ascii=False, indent='    ')


if __name__ == '__main__':
    annotation_file = '../data/animalpose/annotations/animalpose_train.json'
    out_file = '../animalpose_train_custom.json'
    get_custom_json(annotation_file, out_file)
