import torch
import torch.nn as nn

def generate_NN_latent_functions(num_samples, xdim=1, zdim=2, bias=0):
    class NN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 10)
            self.fc4 = nn.Linear(10, output_dim)

            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            with torch.no_grad():
                inp = x
                x1 = torch.relu(self.fc1(x))
                #print("x1:", x1.norm(p = 2, dim = 1))
                x2 = torch.relu(self.fc2(x1))
                #print("x2:", x2.norm(p = 2, dim = 1))
                x3 = torch.relu(self.fc3(x2))
                #print("x3:", x3.norm(p = 2, dim = 1))
                #print("--"*100)
                x4 = (self.fc4(x3))
                x = torch.cat((x4/100, inp), dim = 1)
            return x

    #  weight initialization function
    def weights_init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)
            #nn.init.xavier_normal_(m.weight, gain = 0.5)
            if m.bias is not None:
                #nn.init.normal_(m.bias, mean=0, std=1)
                nn.init.constant_(m.bias, val=1)

    #  neural networks
    networks = []
    for _ in range(num_samples):
        net = NN(xdim, zdim)
        net.apply(weights_init_normal)
        networks.append(net)

    return networks


def find_nns(Y, G, threshold=0.0, disp=False):
    # Y: [1, 1024, 3]
    # G: [20, 1024, 3]
    # threshold: float, the minimum distance must be greater than this value

    # Calculate the pairwise distances
    distances = torch.sum(((Y - G) ** 2), dim=2).mean(dim=1)
    
    # Filter distances greater than the threshold
    filtered_distances = distances[distances > threshold]
    rest_filtered_distances = distances[distances <= threshold]
    
    if len(filtered_distances) > 0:
        # If there are distances greater than the threshold, find the minimum among them
        min_distance, min_idx = torch.min(filtered_distances, dim=0)
    else:
        # If all distances are below the threshold, find the overall minimum
        min_distance, min_idx = torch.min(distances, dim=0)

    if disp:
        print(f"Minimum distance: {min_distance.item()}")
        print(f"Index of minimum distance: {min_idx.item()}")
        print(f"Number of distances below threshold: {len(rest_filtered_distances)}")
        print("---" * 20)
    return min_idx.item()

def diffs(Y, G):
    weighted_diffs = (G - Y)**2
    diffs = torch.sum(weighted_diffs, dim=2)
    return diffs

def f_loss(Y, G):
    diff = diffs(Y,G)
    point_loss_mean = diff.mean(dim=1)
    curve_loss_mean = point_loss_mean.mean(dim=0)
    return curve_loss_mean