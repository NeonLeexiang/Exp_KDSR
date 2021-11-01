#!/bin/bash

#SBATCH -J 31EDl2ll
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH -t 240:00:00
#SBATCH --gres=gpu:1






module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python ../test_conda_env_setting.py


cd ../PISR || exit


#sleep 10h

python step2_train_student.py --config configs/channel_vdsr_32_64/step2_decrease_ende_layerloss_l2.yml

sleep 60

python step2_train_student.py --config configs/channel_vdsr_32_64/step2_decrease_ende_layerloss_ll.yml
