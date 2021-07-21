# Install
**1.**  
git clone

**2.**  
pip install -r requirements.txt

**3.**  
pip uninstall shapely
conda install shapely

**4.**  
cd keypoint_detector
python setup.py develop
cd ../

**5.**  
cd xtcocoapi
python setup.py install
cd ../

# Training
```shell
python keypoint_detector/tools/train.py ${CONFIG_FILE} [optional arguments]  
```

e.g.)  
```shell
python keypoint_detector/tools/train.py keypoint_detector/configs/hrnet_w48_animalpose_256x256.py
```

--validate (strongly recommended): Perform evaluation at every k (default value is 5 epochs during the training.
--work-dir ${WORK_DIR}: Override the working directory specified in the config file.
--resume-from ${CHECKPOINT_FILE}: Resume from a previous checkpoint file.
--gpus ${GPU_NUM}: Number of gpus to use, which is only applicable to non-distributed training.
--seed ${SEED}: Seed id for random state in python, numpy and pytorch to generate random numbers.
--deterministic: If specified, it will set deterministic options for CUDNN backend.
JOB_LAUNCHER: Items for distributed job initialization launcher. Allowed choices are none, pytorch, slurm, mpi. Especially, if set to none, it will test in a non-distributed mode.
LOCAL_RANK: ID for local rank. If not specified, it will be set to 0.
--autoscale-lr: If specified, it will automatically scale lr with the number of gpus by Linear Scaling Rule.
