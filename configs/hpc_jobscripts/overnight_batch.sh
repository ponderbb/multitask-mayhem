#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J 8h_batch 
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s202821@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification if job failed--
#BSUB -Ne
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

# Load the cuda module

# activate environment
source ~/miniconda3/bin/activate
conda activate multitask-mayhem

python src/pipeline/train_model.py -c configs/model-zoo/deeplabv3.yaml
python src/pipeline/train_model.py -c configs/model-zoo/ssdlite.yaml
python src/pipeline/train_model.py -c configs/model-zoo/frcnn.yaml
python src/pipeline/train_model.py -c configs/model-zoo/frcnn-resnet.yaml

# python src/pipeline/train_model.py -c configs/model-zoo/frcnn-hybrid.yaml
# python src/pipeline/train_model.py -c configs/model-zoo/ssdlite-hybrid.yaml


## submit by using: bsub < configs/hpc_jobscripts/overnight_batch.sh 