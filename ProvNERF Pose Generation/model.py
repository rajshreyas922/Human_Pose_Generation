import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(H_theta, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def generate_NN_latent_functions(num_samples, xdim=1, zdim=2, lambda_value=1):
    # Define the neural network class
    class NN(nn.Module):
        
        def __init__(self, input_dim, output_dim):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100, output_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Custom weight initialization function
    def weights_init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Generate and initialize the neural networks
    networks = []
    for _ in range(num_samples):
        net = NN(xdim, zdim)
        net.apply(weights_init_normal)
        networks.append(net)
    
    return networks

def find_nns(Y, G):
    Y = Y.unsqueeze(0)
    result = torch.empty(G.shape[0])
    for i in range(G.shape[0]):
        diffs = (((G[i, :, 0] - Y[0, :, 0]) ** 2 + (G[i, :, 1] - Y[0, :, 1]) ** 2) + (G[i, :, 2] - Y[0, :, 2]) ** 2).sum()
        result[i] = diffs
    return torch.argmin(result).item()

# def f_loss(Y, G, pushing_weight=1.0, pushing_radius=1.0):
#     num_curves = Y.shape[0]
#     total_loss = 0.0
#     pushing_loss = 0.0
    
#     for i in range(num_curves):
#         diffs = (((G[i, :, 0] - Y[i, :, 0]) ** 2 + (G[i, :, 1] - Y[i, :, 1]) ** 2) + 3 * (G[i, :, 2] - Y[i, :, 2]) ** 2)
#         total_loss += diffs.mean()
        
#         num_points = G.shape[1]
#         for j in range(num_points - 1):
#             z_distance = torch.abs(G[i, j, 2] - G[i, j + 1, 2])
#             if z_distance < pushing_radius:
#                 pushing_loss += pushing_weight * (pushing_radius - z_distance) ** 2
    
#     result = (total_loss / num_curves) + (pushing_loss / num_curves)
#     return result

def f_loss(Y, G, pushing_weight=1.0, pushing_radius=1.0):  
    diffs = torch.sqrt((G[:, :, 0] - Y[:, :, 0]) ** 2 + 
             (G[:, :, 1] - Y[:, :, 1]) ** 2 + 
             (G[:, :, 2] - Y[:, :, 2]) ** 2)
    total_loss = diffs.mean()

    return total_loss


def chamfer_distance(Y, G):

    batch_size, num_points_Y, _ = Y.shape
    _, num_points_G, _ = G.shape

    Y_expand = Y.unsqueeze(2).expand(batch_size, num_points_Y, num_points_G, 3)
    G_expand = G.unsqueeze(1).expand(batch_size, num_points_Y, num_points_G, 3)
    distances = torch.norm(Y_expand - G_expand, dim=3)

    min_dist_Y_to_G = torch.min(distances, dim=2)[0]
    min_dist_G_to_Y = torch.min(distances, dim=1)[0]

    chamfer_dist = torch.mean(min_dist_Y_to_G) + torch.mean(min_dist_G_to_Y)
    return chamfer_dist
