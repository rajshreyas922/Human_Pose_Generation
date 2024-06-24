# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur FÃ¶rderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur FÃ¶rderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman
import sys

#Change to your directory for this to work - raj
sys.path.append('/project/6054857/rsp8/STAR')
import matplotlib

from star.pytorch.star import STAR
import numpy as np
from numpy import newaxis
import pickle
import os
import torch

star = STAR(gender='female', num_betas=10)
betas = np.array([
            np.array([ 2.25176191, -3.7883464, 0.46747496, 3.89178988,
                      2.20098416, 0.26102114, -3.07428093, 0.55708514,
                      -3.94442258, -2.88552087])])
num_betas=10

batch_size=1
#m = STAR(gender='male',num_betas=num_betas)

# Zero pose
poses = torch.cuda.FloatTensor(np.zeros((batch_size,72)))
#poses = 2 * np.pi * torch.rand((batch_size, 72), device='cuda')
betas = torch.cuda.FloatTensor(betas)

trans = torch.cuda.FloatTensor(np.zeros((batch_size,3)))
model = star.forward(poses, betas,trans)
shaped = model.v_shaped[-1, :, :]


shaped_np = shaped.cpu().numpy()  

# Create PLY file
ply_header = '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
end_header
'''

with open('human_body_model.ply', 'w') as f:
    f.write(ply_header.format(vertex_count=len(shaped_np)))
    np.savetxt(f, shaped_np, fmt='%f %f %f')