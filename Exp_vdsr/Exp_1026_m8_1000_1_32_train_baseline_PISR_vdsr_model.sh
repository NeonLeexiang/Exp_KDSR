#!/bin/bash

#SBATCH -J 26VTS
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH -t 90:00:00
#SBATCH --gres=gpu:1






module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python ../test_conda_env_setting.py


cd ../PISR || exit

python step1_train_teacher.py --config configs/conv_vdsr_32_64/step1_8.yml


python step2_train_student.py --config configs/conv_vdsr_32_64/step2_8.yml



