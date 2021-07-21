## Install
a. git clone  
```shell
git clone https://github.com/maveltoz/ADK2021.git
```

b. install requirements
```shell
pip install -r requirements.txt
```

c. reinstall shapely
```shell
pip uninstall shapely  
conda install shapely
```

d. setup keypoint_detector
```shell
cd keypoint_detector  
python setup.py develop  
cd ../
```

e. setup xtcocoapi
```shell
cd xtcocoapi  
python setup.py install  
cd ../
```

## Preparation
a. Data Preparation  
Nas -> root -> 연구원자료 -> 김대훈 -> ADK2021 -> data.7z 다운로드 후 ADK2021 폴더에 압축 해제  

b. Weight Preparation  
Nas -> root -> 연구원자료 -> 김대훈 -> ADK2021 -> weights.7z 다운로드 후 ADK2021 폴더에 압축 해제  

- 최종 폴더 구성  

```text
AADK2021
├── data
    │── animalpose
            │-- ...
├── keypoint_detector
├── object_detector
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
run main.py
```

- `img_root(main.py 12 lines)` : inference image root folder
- `ann_root(main.py 13 lines)` : inference image's annotation root folder
- `model(main.py 30 lines)` : init_pose_model(config file, checkpoint weight)
- `vis_pose_result(main.py 59 lines)` : save result images to './results'
- `keypoint_pck_accuracy(main.py 99 lines)` : evaluate PCK
