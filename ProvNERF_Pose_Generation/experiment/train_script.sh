#!/bin/bash
#SBATCH --account=def-keli
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0
#SBATCH --mail-user=rsp8@sfu.ca
#SBATCH --mail-type=ALL

cd ~/$projects/def-keli/rsp8/Human_Pose_Generation/ProvNERF Pose Generation/experiment/
module purge
module load StdEnv/2020
module load python/3.11.5 scipy-stack

virtualenv --no-download ~/py311

source ~/py311/bin/activate

python train.py
