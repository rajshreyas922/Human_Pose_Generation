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

batch_size = 2  # Set your desired batch size here
betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087]),
            1.5*np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
# Generate different betas for each model
betas = torch.cuda.FloatTensor(betas)

# Generate different poses for each model
poses = torch.cuda.FloatTensor(np.zeros((batch_size, 72))) + np.pi
print(poses)
# Generate translations for each model (can be set to zero)
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))

# Forward pass through the STAR model
model = star.forward(poses, betas, trans)
shaped = model.v_shaped

shaped_np = shaped.cpu().numpy()

# Create OBJ files
for i in range(batch_size):
    with open(f'STAR/results/human_body_model_{i + 1}.obj', 'w') as f:
        for vertex in shaped_np[i]:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
