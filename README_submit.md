${PROJECT}  
|-- configs  
|    |-- hrnet.py  
|-- keypoint_detector  
|    |-- mmpose  
|    |    |-- heatmap_focal_loss.py  
|-- tools  
|    |-- autoanchor.py  
|    |-- datasets.py  
|    |-- general.py  
|    |-- google_utils.py  
|    |-- metrics.py  
|    |-- plots.py  
|    |-- torch_utils.py  
|-- README.md  
|-- submit.py  
|-- train.py  
|-- util.py  
|-- outputs  
|    |-- model_final.pth  
|    |-- result.json \


## 주요 코드 설명
- ./configs/hrnet.py : 학습 및 추론에 필요한 설정
- ./preprocessing_data.py : 제공된 train dataset annotation을 코드에 맞는 형식으로 변경  
(새로운 annotation이 ./DATA/train/annotations/animalpose_train.json으로 저장됨)
- ./tools/train.py : 학습을 실행하는 코드
- ./tools/test.py : 추론을 실행하는 코드
- ./keypoint_detector/mmpose/apis/train.py : 키포인트를 학습하는 코드
- ./keypoint_detector/mmpose/apis/test.py : 키포인트를 추론하는 코드
- ./keypoint_detector/mmpose/core/evaluation : validation, test set에 대해 pck, mAP 측정
- ./keypoint_detector/mmpose/core/fp16 : 추론 속도 향상에 사용된 코드
- ./keypoint_detector/mmpose/core/post_processing : 후처리에 사용된 코드
- ./keypoint_detector/mmpose/datasets/datasets : 학습 및 추론에 사용되는 dataset을 정의
- ./keypoint_detector/mmpose/datasets/pipelines : 학습 및 추론에 사용되는 pipeline을 정의
- ./keypoint_detector/mmpose/models/backbones : backbone 모델
- ./keypoint_detector/mmpose/models/detectors : 전체적인 모델
- ./keypoint_detector/mmpose/models/heads : 헤드 모델
- ./keypoint_detector/mmpose/models/losses : 학습에 사용된 loss 모델링


## 학습, 추론 환경
cd /workspace/ADK2021
source /workspace/test/bin/activate

## 학습 데이터셋 annotation의 전처리에 필요한 명령어
$python tools/preprocessing_data.py

## 학습에 필요한 명령어
$python tools/train.py configs/hrnet.py

## 추론에 필요한 명령어
$python tools/test.py configs/hrnet.py weights/best.pth

