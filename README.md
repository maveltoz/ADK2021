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
├── inference_utils
├── keypoint_detector
├── results
├── weights
├── work_dirs
├── xtcocoapi
```

## Training
```shell
python keypoint_detector/tools/train.py ${CONFIG_FILE} [optional arguments]  
```

e.g.)  
```shell
python keypoint_detector/tools/train.py keypoint_detector/configs/hrnet_w48_animalpose_256x256.py
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
python main.py
```

- `img_root(main.py 11 lines)` : inference image root folder
- `ann_root(main.py 12 lines)` : inference image's annotation root folder
- `keypoint_detector_model(main.py 16 lines)` : pretrained weight path
- `eval_pck(main.py 21 lines)` : if True: evaluate PCK
- `visualization_result(main.py 22 lines)` : if True: save result images to './results'
