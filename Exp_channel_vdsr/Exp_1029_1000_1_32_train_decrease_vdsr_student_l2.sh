#!/bin/bash

#SBATCH -J 29l2ll
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

python step3_train_base_model.py --config configs/channel_vdsr_32_64/step4.yml

sleep 60

python step2_train_student.py --config configs/channel_vdsr_32_64/step2_decrease_layerloss_l2.yml

sleep 60

python step2_train_student.py --config configs/channel_vdsr_32_64/step2_decrease_layerloss_ll.yml
