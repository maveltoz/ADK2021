## Install
a. git clone  
```shell
git clone https://github.com/maveltoz/ADK2021.git
```

b. install requirements
```shell
pip install -r requirements.txt
```

c. setup keypoint_detector
```shell
cd keypoint_detector  
python setup.py develop  
cd ../
```

d. setup xtcocoapi
```shell
cd xtcocoapi  
python setup.py install  
cd ../
```

## Preparation
a. Data Preparation  
Nas -> root -> 연구원자료 -> 김대훈 -> ADK2021 -> data.7z 다운로드 후 ADK2021 폴더에 압축 해제  

b. Weight Preparation  
Nas -> root -> 연구원자료 -> 김대훈 -> ADK2021 -> epoch_210.pth 다운로드 후 ADK2021/weights 폴더에 저장  

- 최종 폴더 구성  

```text
ADK2021
├── data
    │── animalpose
            │-- ...
├── keypoint_detector
├── weights
├── xtcocoapi
```

## Training
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]  
```

e.g.)  
```shell
python tools/train.py configs/hrnet.py
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5 epochs during the training.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.
- `--autoscale-lr`: If specified, it will automatically scale lr with the number of gpus by [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

## Inference
```shell
python tools/test.py configs/hrnet.py weights/best.pth --out out.json
```

- `--config` : config.py filename
- `--checkpoint` : checkpoint.pth filename
- `--eval` : if True: evaluate PCK (default=True)
- `--out` : result filename (save result to './result')

## Tensorboard
- config file에 dict(type='TensorboardLoggerHook') 있는지 확인 후 학습( ex> hrnet_w48_256x256.py 26번째 줄 )
- 학습 종료 후 anaconda prompt에서 해당 가상환경 activate
- 해당 프로젝트 폴더로 이동 ( ex> cd ADK2021 )
- `tensorboard --logdir=./work_dirs/hrnet_w48_256x256/tf_logs/` 입력
- 화면에 나오는 주소로 접속( ex> http://localhost:6006 )
- 참고 자료) https://copycoding.tistory.com/88
