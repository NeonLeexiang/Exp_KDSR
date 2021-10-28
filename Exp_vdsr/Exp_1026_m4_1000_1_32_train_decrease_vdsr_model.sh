#!/bin/bash

#SBATCH -J 26VDSA
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -t 90:00:00
#SBATCH --gres=gpu:2






module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python ../test_conda_env_setting.py


cd ../PISR || exit


python step2_train_student.py --config configs/conv_vdsr_32_64/step2_4_decrease.yml

sleep 60

python step2_train_student.py --config configs/conv_vdsr_32_64/step2_4_decrease_layerloss_l1.yml

sleep 60

python step2_train_student.py --config configs/conv_vdsr_32_64/step2_4_decrease_layerloss_l2.yml

sleep 60

python step2_train_student.py --config configs/conv_vdsr_32_64/step2_4_decrease_layerloss_ll.yml
