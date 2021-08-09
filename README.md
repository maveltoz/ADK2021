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
ADK2021 Dataset(save to './data')  

b. Weight Preparation  
Pretrain Weight(save to './weights')  
https://drive.google.com/file/d/1H5LoUbjD8AYBs5pZn2HR4i78NcyGmk8a/view?usp=sharing  

- Folder Configuration  

```text
ADK2021
├── configs
├── data
    │── animalpose
            │-- ...
├── keypoint_detector
├── tools
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

## Pretrain Weight
https://drive.google.com/file/d/1H5LoUbjD8AYBs5pZn2HR4i78NcyGmk8a/view?usp=sharing

## Inference
```shell
python tools/test.py configs/hrnet.py weights/best.pth --out out.json
```

- `--config` : config.py filename
- `--checkpoint` : checkpoint.pth filename
- `--eval` : if True: evaluate PCK (default=True)
- `--out` : result filename (save result to './result')

## Tensorboard
- Check config file. dict(type='TensorboardLoggerHook')
- Activate virtual environment
- Move to project folder ( ex> cd ADK2021 )
- `tensorboard --logdir=./work_dirs/hrnet_w48_256x256/tf_logs/`
- Access to the address on the screen ( ex> http://localhost:6006 )
