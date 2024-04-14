#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.10
module load StdEnv/2020
module load  gcc/9.3.0 arrow/9.0.0 opencv/4.8.0
source ../medsam_xp/bin/activate

python train_one_gpu_slimmedsamlocal.py -task_name MedSlimSAM-ViT-B -model_type vit_p50 -checkpoint /home/xinpeng/SlimSAM/checkpoints/vit_b_slim_step2_.pth -num_epochs 2
