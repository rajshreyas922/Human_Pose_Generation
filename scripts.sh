salloc --time=3:00:0 --ntasks=10 --gres=gpu:v100l:1 --mem=24G --nodes=1

module load StdEnv/2020
module load python/3.11.5 scipy-stack
virtualenv --no-download ~/py311
source ~/py311/bin/activate
pip install --no-index torch torchvision

cd PAPR
python train.py --opt configs/nerfsyn/chair.yml

scancel $(squeue -u $USER -h -o %i)
