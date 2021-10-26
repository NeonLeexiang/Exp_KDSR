# Exp_VDSR_Record

## Baseline

File name: `Exp_1022_1000_1_32_train_vdsr_teacher.sh`
- Sbatch name: 22VT
- Jobid: 71643
- Sbatch time: 
- model name: vdsr_teacher
- output path: /result/conv_vdsr_32_64/vdsr_teacher
- loss: vid_loss
- lr: start->0.001
- config: [step1](../PISR/configs/conv_vdsr_32_64/step1.yml)
- training result: [events.out.tfevents.1634884795.node002](../PISR/results/conv_vdsr_32_64/vdsr_teacher/events.out.tfevents.1634884795.node002)
- `psnr`: `34.6`

File name: `Exp_1022_1000_1_32_train_vdsr_teacher.sh`  
- Sbatch name: 22VS
- model name: vdsr_student
- initialize: vdsr_teacher::all layers
- loss: vid_loss(baseline)
- lr: start->0.001
- config: [step2](../PISR/configs/conv_vdsr_32_64/step2.yml)
- training result:

_some training problem cost by the CUDA memory error, so we need to decrease the m into 8_

File name: `Exp_1025_m8_1000_1_32_train_vdsr_teacher.sh`
- Sbatch name: 25VT8
- Jobid: 72109
- Sbatch time: 1025-19:30
- model name: vdsr_teacher
- output path: /result/conv_vdsr_32_64_m8/vdsr_teacher
- loss: vid_loss
- lr: start->0.001
- config: [step1_8](../PISR/configs/conv_vdsr_32_64/step1_8.yml)
- training result:
- 

---
## Parallel training with 2 GPUS

File name: `Exp_1025_parallel_1000_1_32_train_parallel_vdsr_teacher.sh`
- Sbatch name: 25VTP
- Jobid: 72109
- Sbatch time: 1025-19:30
- model name: parallel_vdsr_teacher
- output path: /result/conv_vdsr_32_64/parallel_vdsr_teacher
- loss: vid_loss
- lr: start->0.001
- config: [step1_16](../PISR/configs/conv_vdsr_32_64/step1_16.yml)
- training result:


---
_due to the CUDA memory error, training the dataset with lower size_

## `m=8` and separate the layer into 4 pieces

File name: `Exp_1025_m8_1000_1_32_train_vdsr_teacher.sh`
- Sbatch name: 25VT8
- Jobid: 72223
- Node: node004
- Sbatch time: 1026-10:35
- model name: vdsr_teacher
- output path: /result/conv_vdsr_32_64_m8/vdsr_teacher
- loss: vid_loss
- lr: start->0.001
- config: [step1_8](../PISR/configs/conv_vdsr_32_64/step1_8.yml)
- training result:
- other: _rebuild the model and set the model layer into 4 pieces for more information._

File name: `Exp_1026_m8_1000_1_32_train_base_vdsr_model.sh`
- Sbatch name: 26VB8
- Jobid: 72227
- Node: node005
- Sbatch time: 1026-10:45
- model name: base_vdsr_student
- output path: /result/conv_vdsr_32_64_m8/base_vdsr_student
- loss: l1loss
- lr: start->0.001
- config: [step4](../PISR/configs/conv_vdsr_32_64/step4.yml)
- training result:
- other: _training the baseline model of vdsr_






