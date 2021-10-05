#!/bin/bash

#SBATCH -J E1002
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1
#SBATCH -w node006


module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python test_conda_env_setting.py


cd PISR || exit

python step1_train_teacher.py --config configs/vdsr_addernet/step1.yml



