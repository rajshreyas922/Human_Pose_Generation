import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(H_theta, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
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

def f_loss(Y, G):
    num_curves = Y.shape[0]
    total_loss = 0.0
    for i in range(num_curves):
        diffs = torch.sum((G[i, :, :] - Y[i, :, :]) ** 2, dim=1)
        total_loss += diffs.mean()

    result = total_loss / num_curves
    return result