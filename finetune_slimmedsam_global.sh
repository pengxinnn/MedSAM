#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.10
module load StdEnv/2020
module load  gcc/9.3.0 arrow/9.0.0 opencv/4.8.0
source ../medsam_xp/bin/activate

sed -i 's/input, approximate=self.approximate/input/g' /home/xinpeng/medsam_xp/lib/python3.10/site-packages/torch/nn/modules/activation.py
python train_one_gpu_slimmedsam_global.py -task_name MedSlimSAM-Global-50-Gaussian -checkpoint /home/xinpeng/SlimSAM/checkpoints/vit_b_medslim_final_step2_0.5_global_gaussian.pth -num_epochs 10 -i data/npy/CT_Abd_train -batch_size 4
