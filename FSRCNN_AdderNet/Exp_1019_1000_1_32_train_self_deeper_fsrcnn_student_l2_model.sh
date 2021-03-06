#!/bin/bash

#SBATCH -J E20AL2
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:1






module load cuda10.2

nvidia-smi
uname -a
cat /proc/version
nvcc -V

python test_conda_env_setting.py


cd PISR || exit

python step2_train_student.py --config configs/self_deeper_fsrcnn_32_64/step2_adder_l2.yml



