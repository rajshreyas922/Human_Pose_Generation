#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:p100:1
#SBATCH --time=15:0:0    
#SBATCH --mail-user=iris_ma@sfu.ca
#SBATCH --mail-type=ALL

cd /home/irisma/projects/def-keli/irisma/human-poses/Human_Pose_Generation/PAPR
module --force purge
module load python/3.9 scipy-stack StdEnv/2020  gcc/9.3.0  cuda/11.4 opencv/4.6.0
source ~/py39/bin/activate

python test.py --opt configs/nerfsyn/chair.yml