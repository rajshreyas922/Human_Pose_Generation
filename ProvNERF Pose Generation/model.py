import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


import torch
import torch.nn.functional as F


class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(H_theta, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def generate_NN_latent_functions(num_samples, xdim=1, zdim=2, lambda_value=1):
    class NN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 50)
            self.fc4 = nn.Linear(50, output_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)*5
            return x

    #  weight initialization function
    def weights_init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain = 0.5)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    #  neural networks
    networks = []
    for _ in range(num_samples):
        net = NN(xdim, zdim)
        net.apply(weights_init_normal)
        networks.append(net)
    return networks


# def find_nns(Y, G):
#     Y_reshaped = Y.view(-1)
#     G_reshaped = G.view(G.shape[0], -1)
#     min_distance = float('inf')
#     min_idx = -1
#     for i in range(G_reshaped.shape[0]):
#         distance = torch.sum((Y_reshaped - G_reshaped[i]) ** 2).item()
#         if distance < min_distance:
#             min_distance = distance
#             min_idx = i
#     return min_idx



# def f_loss(Y, G):
#     weighted_diffs = (G - Y)**2
#     diffs = torch.sum(weighted_diffs, dim=2)
#     total_loss = diffs.mean(dim=1).mean(dim=0)
#     return total_loss

# def chamfer_distance(G, Y):
#     # G: [num_clouds, num_points, 3]
#     # Y: [num_clouds, num_points, 3]

#     # Compute pairwise squared distances between all points in each cloud
#     G_expanded = G[:, :, None, :]  # Shape: [num_clouds, num_points, 1, 3]
#     Y_expanded = Y[:, None, :, :]  # Shape: [num_clouds, 1, num_points, 3]

#     squared_distances = torch.sum((G_expanded - Y_expanded) ** 2, dim=-1)  # Shape: [num_clouds, num_points, num_points]

#     # Find the minimum distance from each point in G to Y
#     min_distances_G_to_Y = torch.min(squared_distances, dim=2)[0]  # Shape: [num_clouds, num_points]

#     # Find the minimum distance from each point in Y to G
#     min_distances_Y_to_G = torch.min(squared_distances, dim=1)[0]  # Shape: [num_clouds, num_points]

#     # Sum the minimum distances and average over all points in the clouds
#     chamfer_dist = min_distances_G_to_Y.mean(dim=1) + min_distances_Y_to_G.mean(dim=1)
#     chamfer_dist = chamfer_dist.mean(dim=0)  # Average over all clouds

#     return chamfer_dist


def find_nns(Y, G):
    #Y: [1, 1024, 3]
    #G: [20, 1024, 3]

    distances = torch.sum(((Y - G) ** 2), dim = 2).mean(dim = 1)
    _, min_idx = torch.min(distances, dim=0)
    return min_idx.item()

def f_loss(Y, G):
    weighted_diffs = (G - Y)**2
    diffs = torch.sum(weighted_diffs, dim=2)
    total_loss = diffs.mean(dim=1).mean(dim=0)
    return total_loss