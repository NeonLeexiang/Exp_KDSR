#!/bin/bash

#SBATCH -J test
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1



module load cuda10.1

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python test_conda_env_setting.py


cd AdderNetCUDA/ || exit

python check.py



