#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.10
module load StdEnv/2020
module load  gcc/9.3.0 arrow/9.0.0 opencv/4.8.0
source ../medsam_xp/bin/activate

python evaluate_global_wo_ft.py --model_path '/home/xinpeng/SlimSAM/checkpoints/vit_b_medslim_final_step2_0.5_local.pth'
# python evaluate_medsam.py --model_path '/home/xinpeng/SlimSAM/checkpoints/medsam_vit_b.pth'
