import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from imle import *
from models import *
import trimesh
from data_process import pos_encoder_decaying
import glob
def plot_generated_curves_grid_2D(
    model, z_in, num_samps, data, out_dir, device, num_points=40, zdim=30, pos_enc_L=4, xdim=1,
    n_rows=10, n_cols=4, xlim=(-7, 7), ylim=(-7, 7), 
    figsize=(20, 50), save_dir='notebook_plots', 
):
    """
    Plots a grid of generated curves alongside reference points.

    Args:
        num_samps (int): Number of samples/generated curves to plot.
        data (torch.Tensor or np.ndarray): Reference points data (shape: [num_points, 2]).
        generated_disp (np.ndarray): Generated curves data (shape: [num_samps, num_points, 2]).
        param_name (str): Name of the parameter or folder to save the plot.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        xlim (tuple): X-axis limits for the plots.
        ylim (tuple): Y-axis limits for the plots.
        figsize (tuple): Size of the figure.
        save_path (str): Path to save the generated plot.
    """
    H_t = model
    Zxs = torch.empty((num_samps, num_points, zdim+int(pos_enc_L*2))).to(device)
    Zs = generate_NN_latent_functions(num_samples=num_samps, xdim=z_in.shape[1], zdim=zdim, bias=1)
    for i, model in enumerate(Zs):
        model = model.to(device)
        z = model(z_in)
        Zxs[i] = z.to(device)
    generated = H_t(Zxs).to(device)



    generated_disp = generated.to(device='cpu').detach().numpy()
    points_disp = data.to(device='cpu').detach().numpy()


    # Create a figure with a specified size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each generated curve
    for i in range(num_samps):
        ax = axes[i]
        
        # Set limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Plot all reference points
        for j in range(data.shape[0]):
            ax.plot(points_disp[j, :, 0], points_disp[j, :, 1], marker='o', color='orange', linestyle='-', linewidth=0.5)
        
        # Plot the generated curve
        ax.plot(generated_disp[i, :, 0], generated_disp[i, :, 1], marker='+', color='blue', linewidth=2)
        
        # Set title for each subplot
        ax.set_title(f"Gen {i+1}")
        
        # Hide axis labels for clarity
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(num_samps, n_rows * n_cols):
        fig.delaxes(axes[i])

    # Set common x and y labels
    fig.text(0.5, 0.04, 'X', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'Y', va='center', rotation='vertical', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Add a main title
    plt.suptitle('Generated Curves Grid', fontsize=20)

    save_path = f"testing_out/{save_dir}/Generated_Curves_Grid.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_generated_curves_3D(H_t, data, z_in, num_samples=15, num_points=576, zdim=3, pos_enc_L=0, device='cpu', save_dir='default'):

    xdim = 2
    
    # Initialize storage for latent representations
    latent_dim = zdim + int(pos_enc_L * 2 * xdim)
    Zxs = torch.empty((num_samples, num_points, latent_dim), device=device)
    
    # Generate latent representations
    Zs = generate_NN_latent_functions(num_samples, xdim=z_in.shape[1], zdim=zdim, bias=1)
    
    # Ensure the save directory exists
    save_path = f"testing_out/{save_dir}"
    os.makedirs(save_path, exist_ok=True)
    
    # Convert all real data to numpy once
    real_data_np = data.cpu().detach().numpy()  # Shape: (batch_size, num_points, 3)
    
    for i, model in enumerate(Zs):
        model = model.to(device)
        Zxs[i] = model(z_in).to(device)

        # FIXED: Proper handling of model output dimensions
        generated = H_t(Zxs[i].unsqueeze(0))  # Shape: (1, num_points, 3)
        generated = generated.squeeze(0)      # Shape: (num_points, 3)
        
        # Convert to numpy
        generated_np = generated.cpu().detach().numpy()  # Shape: (num_points, 3)
        xg, yg, zg = generated_np[:, 0], generated_np[:, 1], generated_np[:, 2]
        
        # Create figure for each sample
        fig, axs = plt.subplots(1, 4, figsize=(18, 6), subplot_kw={'projection': '3d'})
        
        for j, (elev, azim) in enumerate([(20, 30), (40, -60), (60, 120), (90, -60)]):
            ax = axs[j]
            
            # Plot generated sample (more prominent)
            ax.scatter(xg, yg, zg, marker='o', label='Generated', s=2.0, alpha=0.9, c='blue')
            
            # Plot multiple real samples for comparison (less prominent)
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for k in range(min(5, real_data_np.shape[0])):  # Show up to 5 real samples
                x_real, y_real, z_real = real_data_np[k, :, 0], real_data_np[k, :, 1], real_data_np[k, :, 2]
                ax.scatter(x_real, y_real, z_real, marker='x', alpha=0.1, s=0.2, 
                          c=colors[k % len(colors)], label=f'Real {k+1}' if j == 0 and k < 3 else "")
            
            ax.set_title(f'Generated Sample {i + 1} - View {j + 1}')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add legend only to first subplot
            if j == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/Generated_Curve_Sample_{i + 1}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {num_samples} images in {save_path}")


def plot_interpolated_curves_3D(H_t, data, z_in, num_interpolations=10, num_points=576, zdim=2, pos_enc_L=7, device='cuda', save_dir='interpolation'):
    print("pos", pos_enc_L)
    xdim = 2
    # Initialize storage for latent representations
    latent_dim = zdim + int(pos_enc_L * 2 * xdim)
    
    # Generate two base latent functions
    Zs = generate_NN_latent_functions(2, xdim=z_in.shape[1], zdim=zdim, bias=1)
    Z1, Z2 = Zs[0].to(device), Zs[1].to(device)
    
    # Ensure the save directory exists
    save_path = f"testing_out/{save_dir}"
    os.makedirs(save_path, exist_ok=True)
    
    # Function to interpolate between two neural networks
    def interpolate_networks(net1, net2, alpha):
        """Interpolate between two networks' weights with parameter alpha (0=net1, 1=net2)"""
        # Create a new network with the same architecture
        interpolated_net = generate_NN_latent_functions(1, xdim=z_in.shape[1], zdim=zdim, bias=1)[0].to(device)
        
        # Interpolate weights and biases for each layer
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_interp, param_interp) in zip(
                net1.named_parameters(), net2.named_parameters(), interpolated_net.named_parameters()
            ):
                param_interp.data = (1 - alpha) * param1.data + alpha * param2.data
        
        return interpolated_net
    
    # Create interpolation points
    alphas = torch.linspace(0, 1, num_interpolations)
    
    # Store all generated curves for final comparison plot
    all_curves = []
    all_alphas = []
    
    for i, alpha in enumerate(alphas):
        # Create interpolated network
        Z_interp = interpolate_networks(Z1, Z2, alpha.item())
        
        # Generate latent representation and curve
        Zx_interp = Z_interp(z_in).to(device)
        print("Zx_inter", Zx_interp.shape)
        generated = H_t(Zx_interp.unsqueeze(0))  # Shape: (1, num_points, 3)
        generated = generated.squeeze(0)
        # Store for later
        all_curves.append(generated.cpu().detach().numpy())
        all_alphas.append(alpha.item())
        
        # Convert to numpy for individual plots
        generated_np = generated.cpu().detach().numpy()
        real_np = data.cpu().detach().numpy()
        
        xg, yg, zg = generated_np[:, 0], generated_np[:, 1], generated_np[:, 2]
        x_real, y_real, z_real = real_np[:, :, 0], real_np[:, :, 1], real_np[:, :, 2]
        
        # Create individual plot for this interpolation
        fig, axs = plt.subplots(1, 4, figsize=(18, 6), subplot_kw={'projection': '3d'})
        
        for j, (elev, azim) in enumerate([(20, 30), (40, -60), (60, 120), (90, -60)]):
            ax = axs[j]
            ax.scatter(xg, yg, zg, marker='o', label='Generated', s=0.75, c='red')
            #ax.scatter(x_real, y_real, z_real, marker='x', alpha=0.05, s=0.05, label='Real data', c='blue')
            ax.set_title(f'Interpolation α={alpha:.2f} - View {j + 1}')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            if j == 0:  # Add legend only to first subplot
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/Interpolated_Curve_Alpha_{alpha:.2f}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
    
    # Create a summary plot showing all interpolations
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection': '3d'})
    axs = axs.flatten()
    
    # Color map for different interpolations
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_curves)))
    
    for j, (elev, azim) in enumerate([(20, 30), (40, -60), (60, 120), (90, -60)]):
        ax = axs[j]
        
        # Plot real data
        real_np = data.cpu().detach().numpy()
        x_real, y_real, z_real = real_np[:, :, 0], real_np[:, :, 1], real_np[:, :, 2]
        ax.scatter(x_real, y_real, z_real, marker='x', alpha=0.05, s=0.05, label='Real data', c='gray')
        
        # Plot all interpolated curves
        for k, (curve, alpha_val, color) in enumerate(zip(all_curves, all_alphas, colors)):
            xg, yg, zg = curve[:, 0], curve[:, 1], curve[:, 2]
            ax.scatter(xg, yg, zg, marker='o', s=1.5, c=[color], 
                      label=f'α={alpha_val:.2f}' if k % max(1, len(all_curves)//5) == 0 else "", 
                      alpha=0.8)
        
        ax.set_title(f'All Interpolations - View {j + 1}')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if j == 0:  # Add legend only to first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/All_Interpolations_Summary.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved {len(alphas)} interpolated images and 1 summary plot in {save_path}")
    
    return all_curves, all_alphas

    # num_samps = 30
    # Zxs = torch.empty((num_samps, num_points, zdim + int(pos_enc_L*2*xdim))).to(device)
    # Zs = generate_NN_latent_functions(num_samples=num_samps, xdim=z_in.shape[1], zdim=zdim, bias=1)
    # # Create a directory to store the .obj files
    # output_dir = f"{save_path}/generated_objs"
    # os.makedirs(output_dir, exist_ok=True)

    # for i, model in enumerate(Zs):
    #     model = model.to(device)
    #     z = model(z_in)
    #     Zxs[i] = z.to(device)
    #     generated = H_t(Zxs[i]).to(device)  # Shape: (576, 3)

    #     # Extract the generated points as a numpy array
    #     points = generated.detach().cpu().numpy()

    #     # Define the filename for the .obj file
    #     obj_filename = f"{save_path}/generated_objs/generated_points_{i}.obj"

    #     # Write the points to the .obj file
    #     with open(obj_filename, "w") as f:
    #         for point in points:
    #             f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
    #     print(f"Saved {obj_filename}")

# device = 'cuda'
# H_t = H_theta_Res(input_dim=50 + int(7 * 2 * 3), 
#                         output_dim=3,
#                         dropout_rate=0.0).to('cuda')
# num_points = 14400
# file_path = f"training_out/Out_new_model_test_10/H_t_weights.pth"
# H_t.load_state_dict(torch.load(file_path))
# data_dir = '/home/rsp8/scratch/Human_Point_Clouds/'

# x1 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
# x2 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
# grid_x1, grid_x2 = torch.meshgrid((x1, x2), indexing='ij')
# x = torch.stack((grid_x1, grid_x2), dim=-1).reshape(-1, 2).to(device)

# data = []
# obj_files = sorted(glob.glob(os.path.join(data_dir, '*.obj')))
# #print(obj_files)
# for file_path in obj_files[4:14]:
#     mesh = trimesh.load(file_path)
#     vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
#     vertices = vertices - vertices.mean(0)
#     data.append(vertices[0:num_points, :].unsqueeze(0))

# data = torch.cat(data, dim=0).to(device)
# z_in = pos_encoder_decaying(x, L=7).to(device)

# plot_generated_curves_3D(
#     H_t=H_t,
#     z_in=z_in,
#     num_points=num_points,
#     num_samples=40,
#     device='cuda',
#     zdim=50,
#     pos_enc_L=7,
#     save_dir=f'Generations_Model_Test',
#     data=data,
# )