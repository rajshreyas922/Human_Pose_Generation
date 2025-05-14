import torch
import torch.nn as nn

def generate_NN_latent_functions(num_samples, xdim=1, zdim=2, bias=0):
    class NN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 10)
            #self.fc4 = nn.Linear(10, 10)
            self.fc5 = nn.Linear(10, output_dim)

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
                #x4 = torch.relu(self.fc4(x3))
                x5 = self.fc5(x3+x2+x1)
                x = torch.cat((x5/100, inp), dim = 1)
            return x

    #  weight initialization function
    def weights_init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)
            #nn.init.xavier_normal_(m.weight, gain = 0.5)
            if m.bias is not None:
                #nn.init.normal_(m.bias, mean=0, std=2)
                nn.init.constant_(m.bias, val=1)

    #  neural networks
    networks = []
    for _ in range(num_samples):
        net = NN(xdim, zdim)
        net.apply(weights_init_normal)
        networks.append(net)

    return networks


# def find_nns(Y, G, threshold=0.0, disp=False):
#     # Y: [1, 1024, 3]
#     # G: [20, 1024, 3]
#     # threshold: float, the minimum distance must be greater than this value

#     # Calculate the pairwise distances
#     distances = torch.sum(((Y - G) ** 2), dim=2).mean(dim=1)
    
#     # Filter distances greater than the threshold
#     filtered_distances = distances[distances > threshold]
#     rest_filtered_distances = distances[distances <= threshold]
    
#     if len(filtered_distances) > 0:
#         # If there are distances greater than the threshold, find the minimum among them
#         min_distance, min_idx = torch.min(filtered_distances, dim=0)
#     else:
#         # If all distances are below the threshold, find the overall minimum
#         min_distance, min_idx = torch.min(distances, dim=0)

#     if disp:
#         print(f"Minimum distance: {min_distance.item()}")
#         print(f"Index of minimum distance: {min_idx.item()}")
#         print(f"Number of distances below threshold: {len(rest_filtered_distances)}")
#         print("---" * 20)
#     return min_idx.item()

def find_nns(Y, G, threshold=0.0, disp=False):
    Y_expanded = Y.expand(G.size(0), -1, -1)  # (B, N, D)

    
    d = torch.cdist(G, Y_expanded, p=2)**2  # (B, N, N)

    part1 = d.min(dim=2)[0].mean(dim=1)  # G -> Y
    part2 = d.min(dim=1)[0].mean(dim=1)  # Y -> G
    distances = (part1 + part2)

    mask = distances > threshold
    valid_indices = mask.nonzero().squeeze(1)

    if valid_indices.numel() > 0:
        candidate_distances = distances[valid_indices]
        min_val, local_idx = torch.min(candidate_distances, dim=0)
        min_idx = valid_indices[local_idx].item()
    else:
        min_val, min_idx = torch.min(distances, dim=0)
        min_idx = min_idx.item()

    if disp:
        print(f"Minimum Chamfer distance: {min_val.item()}")
        print(f"Index of minimum: {min_idx}")
        print(f"Number of losses over threshold: {valid_indices.shape}")
        print("---" * 20)

    return min_idx, min_val.item()

# def find_nns(Y, G, threshold=0.0, disp=False, num_samples=256):
#     """
#     Y: (1, N, D) - reference curve
#     G: (B, N, D) - generated candidate curves
#     """
#     Y = Y.unsqueeze(0)
#     def subsample(tensor, k):
#         idx = torch.randperm(tensor.size(1), device=tensor.device)[:k]
#         return tensor[:, idx, :]

#     Y_sub = subsample(Y, num_samples)  # (1, k, D)
#     G_sub = subsample(G, num_samples)  # (B, k, D)

#     # Efficient L2^2 distance
#     B, k, D = G_sub.shape
#     Y_rep = Y_sub.repeat(B, 1, 1)  # (B, k, D)

#     G_sq = G_sub.pow(2).sum(dim=2, keepdim=True)      # (B, k, 1)
#     Y_sq = Y_rep.pow(2).sum(dim=2).unsqueeze(1)       # (B, 1, k)
#     inner = torch.bmm(G_sub, Y_rep.transpose(1, 2))   # (B, k, k)
#     d = G_sq - 2 * inner + Y_sq                       # (B, k, k)

#     # Chamfer distance (symmetric)
#     part1 = d.min(dim=2)[0].mean(dim=1)  # G -> Y
#     part2 = d.min(dim=1)[0].mean(dim=1)  # Y -> G
#     distances = part1 + part2            # (B,)

#     # Thresholding
#     mask = distances > threshold
#     valid_indices = mask.nonzero(as_tuple=True)[0]

#     if valid_indices.numel() > 0:
#         candidate_distances = distances[valid_indices]
#         min_val, local_idx = torch.min(candidate_distances, dim=0)
#         min_idx = valid_indices[local_idx].item()
#     else:
#         min_val, min_idx = torch.min(distances, dim=0)
#         min_idx = min_idx.item()

#     if disp:
#         print(f"Minimum Chamfer distance: {min_val.item()}")
#         print(f"Index of minimum: {min_idx}")
#         print(f"Number of losses over threshold: {valid_indices.shape}")
#         print("---" * 20)

#     return min_idx, min_val.item()


def f_loss(Y, G):
    d = torch.cdist(Y, G, p=2)**2
    diff = d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)
    return diff.mean()


# def knn(x, k):
#     """Find k nearest neighbors for each point in x."""
#     # x shape: (batch_size, num_points, num_dims)
#     inner = -2 * torch.matmul(x, x.transpose(2, 1))
#     xx = torch.sum(x**2, dim=-1, keepdim=True)
#     # pairwise_distance shape: (batch_size, num_points, num_points)
#     pairwise_distance = xx + inner + xx.transpose(2, 1)
#     # Find k+1 neighbors because the point itself is included
#     # indices shape: (batch_size, num_points, k)
#     # Exclude self (index 0) which has distance 0
#     idx = pairwise_distance.topk(k=k+1, dim=-1, largest=False)[1][:, :, 1:]
#     return idx

# def gather_neighbors(x, idx):
#     """Gather neighbor points based on indices."""
#     # x shape: (batch_size, num_points, num_dims)
#     # idx shape: (batch_size, num_points, k)
#     batch_size, num_points, k = idx.size()
#     num_dims = x.size(2)

#     # Create indices for batch dimension
#     batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, num_points, k)
#     # Use advanced indexing to gather neighbors
#     # neighbors shape: (batch_size, num_points, k, num_dims)
#     neighbors = x[batch_idx, idx, :]
#     return neighbors

# def laplacian_loss(point_cloud, k=10):
#     """
#     Computes the Laplacian smoothing loss for the input point cloud.
#     Encourages each point to be near the centroid of its k neighbors within the cloud.

#     Args:
#         point_cloud (torch.Tensor): The point cloud to smooth (batch_size, num_points, num_dims).
#         k (int): Number of nearest neighbors to consider.

#     Returns:
#         torch.Tensor: Laplacian loss scalar.
#     """
#     # point_cloud shape: (B, N, D) e.g., (16, 1024, 3)
#     idx = knn(point_cloud, k) # (B, N, k)
#     neighbors = gather_neighbors(point_cloud, idx) # (B, N, k, D)

#     # Calculate the centroid of neighbors
#     centroid = torch.mean(neighbors, dim=2) # (B, N, D)

#     # Calculate the squared distance between each point and its neighbor centroid
#     laplacian_diff = point_cloud - centroid # (B, N, D)
#     laplacian_norm_sq = torch.sum(laplacian_diff**2, dim=2) # (B, N)

#     # Average over points and then over batch
#     loss = torch.mean(laplacian_norm_sq, dim=1) # (B,)
#     return loss.mean() # scalar

# # --- Original Chamfer Loss (using Y=GT, G=Pred terminology) ---
# def chamfer_loss_f(Y_gt, G_pred):
#     """
#     Computes Chamfer Distance between Y_gt (ground truth) and G_pred (prediction).

#     Args:
#         Y_gt (torch.Tensor): Ground truth point cloud (B, N_y, D).
#         G_pred (torch.Tensor): Predicted point cloud (B, N_g, D).

#     Returns:
#         torch.Tensor: Chamfer distance scalar.
#     """
#     # d shape: (B, N_y, N_g)
#     d = torch.cdist(Y_gt, G_pred, p=2)**2
#     # Find nearest in G_pred for each point in Y_gt -> how well Y is covered by G
#     min_dist_y_g = d.min(dim=2)[0] # (B, N_y)
#     # Find nearest in Y_gt for each point in G_pred -> how well G covers Y
#     min_dist_g_y = d.min(dim=1)[0] # (B, N_g)
#     # Original definition sums the means
#     loss_term1 = min_dist_y_g.mean(dim=1) # (B,)
#     loss_term2 = min_dist_g_y.mean(dim=1) # (B,)
#     diff = loss_term1 + loss_term2 # (B,)
#     return diff.mean() # scalar

# # --- Combined Loss ---
# def combined_loss_smooth_generated(Y_gt, G_pred, k=10, lambda_laplacian=0.1):
#     """
#     Combines Chamfer distance (fidelity) with Laplacian smoothing regularization
#     applied to the generated point cloud G_pred.

#     Args:
#         Y_gt (torch.Tensor): Ground truth point cloud (B, N_y, D).
#         G_pred (torch.Tensor): Generated/Predicted point cloud (B, N_g, D).
#         k (int): Number of neighbors for Laplacian loss.
#         lambda_laplacian (float): Weighting factor for the Laplacian loss.

#     Returns:
#         torch.Tensor: Combined loss scalar.
#     """
#     # Fidelity term: How close is the prediction G to the ground truth Y?
#     fidelity_loss = chamfer_loss_f(Y_gt, G_pred)

#     # Regularization term: How smooth is the prediction G internally?
#     smoothing_loss = laplacian_loss(G_pred, k) # Apply to G_pred!

#     # Combine the losses
#     total_loss = fidelity_loss + lambda_laplacian * smoothing_loss
#     return total_loss


# def diffs(Y, G):

#     weighted_diffs = (G - Y)**2
#     diffs = torch.sum(weighted_diffs, dim=2)
#     return diffs

# def f_loss(Y, G):
#     diff = diffs(Y,G)
#     point_loss_mean = diff.mean(dim=1)
#     curve_loss_mean = point_loss_mean.mean(dim=0)
#     return curve_loss_mean