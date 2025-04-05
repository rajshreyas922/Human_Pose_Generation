#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --time=15:0:0    
#SBATCH --mail-user=iris_ma@sfu.ca
#SBATCH --mail-type=ALL

cd /home/irisma/projects/def-keli/irisma/human-poses/Human_Pose_Generation/4D-Humans
module --force purge
module load python/3.9 scipy-stack/2022a StdEnv/2020  gcc/9.3.0  cuda/11.4 opencv/4.6.0
source ~/py39/bin/activate

python demo.py     --img_folder iris-data/images     --out_folder iris_out     --batch_size=48 --side_view --save_mesh --full_frame