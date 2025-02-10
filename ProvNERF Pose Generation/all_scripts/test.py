import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from imle import *

def plot_generated_curves_grid(
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

