# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
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
sys.path.append('STAR')
from star.tf.star import STAR
import tensorflow as tf
import numpy as np



def idx_to_theta(index):
    return [index*3 ,index*3 + 1,index*3 + 2]


batch_size = 1
gender = 'male'
star = STAR()
trans = tf.convert_to_tensor(np.zeros((batch_size, 3)), dtype=tf.float32)

pose = np.zeros((batch_size, 72), dtype= np.float32)




pose[0, 3] = 0 
pose[0, 4] = 0 
pose[0, 5] = 0 

shoulder_left = idx_to_theta(16)
shoulder_right = idx_to_theta(17)

pose[0, shoulder_left[1]] = np.pi/4
pose[0, shoulder_right[1]] = np.pi/4

pose[0, shoulder_left[2]] = np.pi/4
pose[0, shoulder_right[2]] = -np.pi/4

leg_left = idx_to_theta(1)
leg_right = idx_to_theta(2)

pose[0, leg_left[0]] = np.pi/4
pose[0, leg_right[0]] = -np.pi/4



pose = tf.convert_to_tensor(pose)


betas = tf.constant(np.random.randn(batch_size, 10)*0, dtype=tf.float32) 

# Forward pass through the STAR model
result = star(pose, betas, trans)
print(result)

# Create OBJ files
for i in range(batch_size):
    with open(f'STAR/results/human_body_model_tf_{i + 1}.obj', 'w') as f:
        for vertex in result[0]:  # Assuming all results are identical for each batch in this example
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

