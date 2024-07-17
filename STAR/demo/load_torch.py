import sys

# Change to your directory for this to work - raj
sys.path.append('STAR')
import matplotlib

from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

star = STAR(gender='neutral', num_betas=10)

batch_size = 3  # Set your desired batch size here
betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
betas = torch.cuda.FloatTensor(np.random.rand(batch_size, 10))*3
poses = np.pi * torch.rand((batch_size, 72), device='cuda')  # 24 joints, 3 angles each

trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))

model = star.forward(poses, betas, trans)
shaped = model.v_shaped

shaped_np = shaped.cpu().numpy()

for i in range(batch_size):
    with open(f'STAR\\results/human_body_model_{i + 1}.obj', 'w') as f:
        for vertex in shaped_np[i]:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
