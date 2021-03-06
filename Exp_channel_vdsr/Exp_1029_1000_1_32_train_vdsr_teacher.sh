#!/bin/bash

#SBATCH -J 29VTBaseline
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH -t 150:00:00
#SBATCH --gres=gpu:1






module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python ../test_conda_env_setting.py


cd ../PISR || exit

python step1_train_teacher.py --config configs/channel_vdsr_32_64/step1.yml

sleep 60

python step2_train_student.py --config configs/channel_vdsr_32_64/step2.yml

