import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os
import argparse

from generate import plot_generated_curves_3D, plot_generated_curves_grid_2D, plot_interpolated_curves_3D
from imle import generate_NN_latent_functions, find_nns, f_loss
from models import *
from data_process import *

from misc import *
import trimesh
import glob
import time

def generate_new_samples_during_training(H_t, z_in, epoch, out_dir, device, zdim, pos_enc_L, xdim, num_samples=20, data=None):
    """
    Generate completely new point cloud samples by sampling new latent functions during training.
    Similar to plot_generated_curves_3D but designed for periodic generation during training.
    """
    # Create samples directory
    samples_dir = f'training_out/{out_dir}/samples'
    os.makedirs(samples_dir, exist_ok=True)
    
    H_t.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Initialize storage for latent representations
        num_points = z_in.shape[0]
        latent_dim = zdim + int(pos_enc_L * 2 * xdim)
        Zxs = torch.empty((num_samples, num_points, latent_dim), device=device)
        
        # Generate NEW latent functions (this is the key difference from your training loop)
        Zs = generate_NN_latent_functions(num_samples, xdim=z_in.shape[1], zdim=zdim, bias=1)
        
        generated_samples = []
        
        # Generate samples from new latent functions
        for i, model in enumerate(Zs):
            model = model.to(device)
            Zxs[i] = model(z_in).to(device)
            
            # Generate point cloud from latent representation
            generated = H_t(Zxs[i].unsqueeze(0))  # Shape: (1, num_points, 3)
            generated = generated.squeeze(0)      # Shape: (num_points, 3)
            
            generated_samples.append(generated.cpu().numpy())
        
        
        # Create visualization if 3D point clouds
        if generated_samples[0].shape[1] == 3:  # 3D point clouds
            fig = plt.figure(figsize=(20, 16))
            
            # Plot samples in a 4x5 grid
            for i in range(min(20, len(generated_samples))):
                ax = fig.add_subplot(4, 5, i+1, projection='3d')
                points = generated_samples[i]
                
                # Plot generated sample
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, alpha=0.8, c='blue')
                ax.set_title(f'New Sample {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y') 
                ax.set_zlabel('Z')
                
                # Set equal aspect ratio
                max_range = np.array([points[:,0].max()-points[:,0].min(),
                                    points[:,1].max()-points[:,1].min(),
                                    points[:,2].max()-points[:,2].min()]).max() / 2.0
                mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
                mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
                mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                # Set viewing angle
                ax.view_init(elev=20, azim=30)
            
            plt.tight_layout()
            plt.savefig(f'{samples_dir}/new_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            
        print(f"Generated and saved {len(generated_samples)} NEW samples at epoch {epoch}")
    
    H_t.train()  # Set back to training mode
    return generated_samples


def create_detailed_sample_plots(generated_samples, real_data, epoch, samples_dir, num_samples=5):
    """
    Create detailed plots similar to plot_generated_curves_3D for a few samples.
    """
    if real_data is not None:
        real_data_np = real_data.cpu().detach().numpy()
    
    for i in range(min(num_samples, len(generated_samples))):
        generated_np = generated_samples[i]
        xg, yg, zg = generated_np[:, 0], generated_np[:, 1], generated_np[:, 2]
        
        # Create figure with multiple views
        fig, axs = plt.subplots(1, 4, figsize=(18, 6), subplot_kw={'projection': '3d'})
        
        for j, (elev, azim) in enumerate([(20, 30), (40, -60), (60, 120), (90, -60)]):
            ax = axs[j]
            
            # Plot generated sample (more prominent)
            ax.scatter(xg, yg, zg, marker='o', label='Generated', s=2.0, alpha=0.9, c='blue')
            
            # Plot real samples for comparison if available
            if real_data is not None:
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for k in range(min(3, real_data_np.shape[0])):  # Show up to 3 real samples
                    x_real, y_real, z_real = real_data_np[k, :, 0], real_data_np[k, :, 1], real_data_np[k, :, 2]
                    ax.scatter(x_real, y_real, z_real, marker='x', alpha=0.15, s=0.5, 
                              c=colors[k % len(colors)], label=f'Real {k+1}' if j == 0 and k < 3 else "")
            
            ax.set_title(f'New Sample {i + 1} - View {j + 1} (Epoch {epoch})')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if j == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{samples_dir}/detailed_new_sample_{i + 1}_epoch_{epoch}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def generate_fixed_latent_functions(z_in, zdim, device, num_samples=10, save_dir='training_out', out_dir=''):
    """
    Generate a fixed set of latent functions at the start of training to track learning progression.
    """
    # Create directory for fixed latent tracking
    fixed_latent_dir = f'{save_dir}/{out_dir}/fixed_latent_tracking'
    os.makedirs(fixed_latent_dir, exist_ok=True)
    
    # Generate fixed latent functions
    print(f"Generating {num_samples} fixed latent functions for tracking...")
    fixed_Zs = generate_NN_latent_functions(num_samples, xdim=z_in.shape[1], zdim=zdim, bias=1)
    
    # Pre-compute latent representations (these will stay constant)
    num_points = z_in.shape[0]
    latent_dim = zdim + z_in.shape[1]  # This should match your actual latent dimension
    fixed_latent_representations = torch.empty((num_samples, num_points, latent_dim), device=device)
    
    for i, model in enumerate(fixed_Zs):
        model = model.to(device)
        with torch.no_grad():
            fixed_latent_representations[i] = model(z_in).to(device)
    
    # Save the fixed latent representations
    torch.save(fixed_latent_representations, f'{fixed_latent_dir}/fixed_latent_representations.pt')
    
    # Save metadata about the latent functions instead of the models themselves
    metadata = {
        'num_samples': num_samples,
        'zdim': zdim,
        'num_points': num_points,
        'latent_dim': latent_dim,
        'z_in_shape': z_in.shape
    }
    torch.save(metadata, f'{fixed_latent_dir}/fixed_latent_metadata.pt')
    
    print(f"Fixed latent representations and metadata saved to {fixed_latent_dir}")
    return fixed_latent_representations


def generate_from_fixed_latents(H_t, fixed_latent_representations, epoch, out_dir, data=None):
    """
    Generate point clouds from the same fixed latent functions to track learning progression.
    """
    # Create directory for this epoch's results
    fixed_latent_dir = f'training_out/{out_dir}/fixed_latent_tracking'
    epoch_dir = fixed_latent_dir
    os.makedirs(epoch_dir, exist_ok=True)
    
    H_t.eval()
    with torch.no_grad():
        generated_samples = []
        num_samples = fixed_latent_representations.shape[0]
        
        # Generate point clouds from fixed latent representations
        for i in range(num_samples):
            # Use the same latent representation as always
            latent_rep = fixed_latent_representations[i].unsqueeze(0)  # Add batch dimension
            
            # Generate point cloud with current H_t model
            generated = H_t(latent_rep)  # Shape: (1, num_points, 3)
            generated = generated.squeeze(0)  # Shape: (num_points, 3)
            
            generated_samples.append(generated.cpu().numpy())
        

        # Create visualization
        if generated_samples[0].shape[1] == 3:  # 3D point clouds
            fig = plt.figure(figsize=(20, 12))
            
            # Calculate grid size
            n_samples = len(generated_samples)
            n_cols = min(5, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            for i in range(n_samples):
                ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
                points = generated_samples[i]
                
                # Plot generated sample
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, alpha=0.8, c='blue')
                ax.set_title(f'Fixed Latent {i+1}\nEpoch {epoch}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set equal aspect ratio
                max_range = np.array([points[:,0].max()-points[:,0].min(),
                                    points[:,1].max()-points[:,1].min(),
                                    points[:,2].max()-points[:,2].min()]).max() / 2.0
                if max_range > 0:
                    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
                    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
                    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                # Set viewing angle
                ax.view_init(elev=20, azim=30)
            
            plt.tight_layout()
            plt.savefig(f'{epoch_dir}/fixed_latent_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"Generated samples from fixed latents at epoch {epoch}")
    
    H_t.train()
    return generated_samples

def create_individual_latent_evolution_plots(out_dir, plot_epoch, epochs):
    """
    Create individual plots showing how each fixed latent function evolves over training.
    """
    fixed_latent_dir = f'training_out/{out_dir}/fixed_latent_tracking'
    
    # Load all epoch data
    epoch_dirs = sorted([d for d in os.listdir(fixed_latent_dir) 
                        if d.startswith('epoch_') and os.path.isdir(f'{fixed_latent_dir}/{d}')])
    
    if len(epoch_dirs) < 2:
        return
    
    # Load data for each epoch
    all_epoch_data = {}
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.split('_')[1])
        npz_files = glob.glob(f'{fixed_latent_dir}/{epoch_dir}/*.npz')
        if npz_files:
            data = np.load(npz_files[0])
            all_epoch_data[epoch_num] = [data[f'arr_{i}'] for i in range(len(data.files))]
    
    if not all_epoch_data:
        return
    
    # Create evolution plots for each latent function
    num_latents = len(next(iter(all_epoch_data.values())))
    epochs_list = sorted(all_epoch_data.keys())
    
    for latent_idx in range(num_latents):
        fig = plt.figure(figsize=(20, 4))
        
        n_epochs_to_show = min(6, len(epochs_list))
        epoch_indices = np.linspace(0, len(epochs_list)-1, n_epochs_to_show, dtype=int)
        
        for i, epoch_idx in enumerate(epoch_indices):
            epoch = epochs_list[epoch_idx]
            points = all_epoch_data[epoch][latent_idx]
            
            ax = fig.add_subplot(1, n_epochs_to_show, i+1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, alpha=0.8, c='blue')
            ax.set_title(f'Latent {latent_idx+1}\nEpoch {epoch}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=30)
            
            # Set equal aspect ratio
            max_range = np.array([points[:,0].max()-points[:,0].min(),
                                points[:,1].max()-points[:,1].min(),
                                points[:,2].max()-points[:,2].min()]).max() / 2.0
            if max_range > 0:
                mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
                mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
                mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(f'{fixed_latent_dir}/latent_{latent_idx+1}_evolution.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Created individual latent evolution plots in {fixed_latent_dir}")


def train(
        H_t,
        optimizer,
        out_dir,
        device,
        epochs=10000,
        staleness=5,
        num_Z_samples=70,
        num_points=40,
        xdim=1,
        zdim=30,
        pos_enc_L=10,
        plot_epoch=250,
        perturb_scale=0.97,
        threshold=0.0,
        data_dir=None,
        batch_size=500,
        clip_grad_norm=1.0,  # Gradient clipping value
        lr_scheduler=None,
        track_fixed_latents=True,  # Add this
        num_fixed_latents=10       # Add this
):
    # Start measuring total training time
    total_training_start = time.time()
    
    grad_norms = []
    param_norms = []
    losses = []
    best_loss = float('inf')
    
    # Create output directory
    os.makedirs('training_out/' + out_dir, exist_ok=True)
    
    # Prepare the data BEFORE the training loop (moved outside)
    data_prep_start = time.time()
    if xdim == 1:
        x = torch.linspace(-0.05, 0.05, num_points).to(device).unsqueeze(1)
        data = generate_data(num_points).to(device)
    else:
        x1 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        x2 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        grid_x1, grid_x2 = torch.meshgrid((x1, x2), indexing='ij')
        x = torch.stack((grid_x1, grid_x2), dim=-1).reshape(-1, 2).to(device)
        
        if data_dir is None:
            data = generate_3D_data(int(np.sqrt(num_points))).to(device)[:, 0:num_points, :]
        else:
            data = []
            obj_files = sorted(glob.glob(os.path.join(data_dir, '*.obj')))
            for file_path in obj_files[4:34]:
                mesh = trimesh.load(file_path)
                vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                vertices = vertices - vertices.mean(0)
                data.append(vertices[0:num_points, :].unsqueeze(0))

            data = torch.cat(data, dim=0).to(device)
    
    # Pre-compute positional encoding once
    z_in = pos_encoder_decaying(x, L=pos_enc_L).to(device)
    data_prep_end = time.time()
    print(f"Data preparation time: {data_prep_end - data_prep_start:.4f}s")
    
    # Prepare batch indices once
    indices = torch.randperm(num_points)
    batches = torch.split(indices, batch_size)  # Automatically handles remainders
    
    # Pre-allocate a fixed tensor for stored transformed points (more efficient than list)
    num_batches = len(batches)
    batch_dims = [len(batch) for batch in batches]
    max_batch_size = max(batch_dims)
    
    # Initialize timing variables
    total_time_z_generation = 0.0
    total_time_perturbation = 0.0
    total_time_reuse = 0.0
    total_time_forward = 0.0
    total_time_backward = 0.0
    
    # Create an array to store IMLE transformed points more efficiently
    stored_imle_transformed_points = {}
    
    # Save initial model checkpoint
    initial_model_path = f'training_out/{out_dir}/H_t_initial.pth'
    torch.save(H_t.state_dict(), initial_model_path)
    fixed_latent_representations = None
    if track_fixed_latents:
        fixed_latent_representations = generate_fixed_latent_functions(
            z_in=z_in,
            zdim=zdim,
            device=device,
            num_samples=num_fixed_latents,
            save_dir='training_out',
            out_dir=out_dir
        )

    # Training loop
    for e in tqdm(range(epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        
        # Clear stored points if staleness period is reached
        if e % staleness == 0:
            stored_imle_transformed_points = {}
        
        # Process each batch
        for batch_idx, batch_indices in enumerate(batches):
            current_batch_size = len(batch_indices)
            data_batch = data[:, batch_indices, :]
            
            # Generate new latent functions if staleness period is reached
            if e % staleness == 0:
                z_gen_start = time.time()
                
                # Generate latent functions (with torch.no_grad for efficiency)
                with torch.no_grad():
                    Zs = generate_NN_latent_functions(
                        num_samples=num_Z_samples,
                        xdim=z_in.shape[1],
                        zdim=zdim,
                        bias=1
                    )
                    
                    # Pre-allocate tensor for efficiency
                    Zxs = torch.empty((num_Z_samples, current_batch_size, zdim + int(pos_enc_L * 2 * xdim)), 
                                      device=device)
                    
                    for i, model in enumerate(Zs):
                        model = model.to(device)
                        z = model(z_in[batch_indices])
                        Zxs[i] = z
                    
                    # Generate and find nearest neighbors
                    generated = H_t(Zxs)
                    imle_nns = [find_nns(d, generated, threshold=threshold) for d in data_batch]
                
                z_gen_end = time.time()
                total_time_z_generation += z_gen_end - z_gen_start
                
                # Perturb the selected models
                perturb_start = time.time()
                imle_transformed_points = torch.empty((data.shape[0], current_batch_size, 
                                                     zdim + int(pos_enc_L * 2 * xdim)), 
                                                    device=device)
                
                for i, (nn_idx, _) in enumerate(imle_nns):
                    original_model = Zs[nn_idx]
                    perturbed_model = copy.deepcopy(original_model).to(device)
                    
                    # Perturb parameters
                    with torch.no_grad():
                        for param in perturbed_model.parameters():
                            param.data += torch.randn_like(param) * perturb_scale
                        z_perturbed = perturbed_model(z_in[batch_indices])
                    
                    imle_transformed_points[i] = z_perturbed
                
                # Store for future use (more efficient with dictionary by batch_idx)
                stored_imle_transformed_points[batch_idx] = imle_transformed_points
                
                perturb_end = time.time()
                total_time_perturbation += perturb_end - perturb_start
            
            else:
                # Reuse previously computed points
                reuse_start = time.time()
                imle_transformed_points = stored_imle_transformed_points[batch_idx]
                reuse_end = time.time()
                total_time_reuse += reuse_end - reuse_start
            

            optimizer.zero_grad(set_to_none=True)  # More efficient than just zero_grad()
            
            forward_start = time.time()
            # Standard training path
            outputs = H_t(imle_transformed_points)
            loss = f_loss(data_batch, outputs)
            loss.backward()
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(H_t.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            forward_end = time.time()
            total_time_forward += forward_end - forward_start
            
            epoch_loss += loss.item()
        
        # Update learning rate if scheduler provided
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Log loss (use log10 for better scale)
        avg_loss = np.log10(epoch_loss)
        losses.append(avg_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(H_t.state_dict(), f'training_out/{out_dir}/H_t_best.pth')
        
        # Periodically save checkpoints
        if e % 1000 == 0 and e > 0:
            torch.save({
                'epoch': e,
                'model_state_dict': H_t.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'training_out/{out_dir}/H_t_checkpoint_{e}.pth')
        
        # Plot results at specified intervals
        if e % plot_epoch == 0 or e == epochs-1:

            with torch.no_grad():  # Ensure we don't track gradients during plotting
                plot_results(
                    data.cpu().numpy(),
                    outputs.detach().cpu().numpy(),
                    losses,
                    e,
                    epochs,
                    out_dir
                )
                if e > 0:
                    generate_new_samples_during_training(
                        H_t=H_t,
                        z_in=z_in,
                        epoch=e,
                        out_dir=out_dir,
                        device=device,
                        zdim=zdim,
                        pos_enc_L=pos_enc_L,
                        xdim=xdim,
                        num_samples=20,
                        data=data
                    )
                
                # Track how fixed latent functions evolve
                if track_fixed_latents and fixed_latent_representations is not None:
                    generate_from_fixed_latents(
                        H_t=H_t,
                        fixed_latent_representations=fixed_latent_representations,
                        epoch=e,
                        out_dir=out_dir,
                        data=data
                    )
        
        # Compute epoch time for logging
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        
        # Print status every 100 epochs
        if e % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {e}/{epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.4f}s, LR: {current_lr:.6f}")

    # End of training timing
    total_training_end = time.time()
    total_training_time = total_training_end - total_training_start
    
    # Final model save
    torch.save(H_t.state_dict(), f'training_out/{out_dir}/H_t_final.pth')
    
    print("\n==== Timing Summary ====")
    print(f"Total training time: {total_training_time:.4f} s")
    print(f"Total time spent generating latent functions (Zs): {total_time_z_generation:.4f} s")
    print(f"Total time spent perturbing and selecting IMLE points: {total_time_perturbation:.4f} s")
    print(f"Total time spent reusing stored IMLE points: {total_time_reuse:.4f} s")
    print(f"Total time spent in forward/backward passes: {total_time_forward:.4f} s")
    print(f"Average time per epoch: {total_training_time/epochs:.4f} s")
    
    # Save timing information to a file
    timing_file = os.path.join('training_out', out_dir, 'timing_info.txt')
    with open(timing_file, 'w') as f:
        f.write(f"Total training time: {total_training_time:.4f} s\n")
        f.write(f"Total time spent generating latent functions (Zs): {total_time_z_generation:.4f} s\n")
        f.write(f"Total time spent perturbing and selecting IMLE points: {total_time_perturbation:.4f} s\n")
        f.write(f"Total time spent reusing stored IMLE points: {total_time_reuse:.4f} s\n")
        f.write(f"Total time spent in forward/backward passes: {total_time_forward:.4f} s\n")
        f.write(f"Average time per epoch: {total_training_time/epochs:.4f} s\n")
        f.write(f"Final loss value: {avg_loss}\n")
        f.write(f"Best loss value: {np.log10(best_loss)}\n")

    return H_t, grad_norms, param_norms, losses


def plot_results(data, outputs, losses, epoch, total_epochs, out_dir):
    # Create output directory if it doesn't exist
    os.makedirs(f'training_out/{out_dir}', exist_ok=True)


    plt.figure(figsize=(20, 16))  
    
    for i in range(10):
        plt.subplot(2, 5, i+1)  # 2 rows, 5 columns, position i+1
        plt.scatter(data[i, :, 0], data[i, :, 1], 
                    c='blue', label='Real', s=3, alpha=0.5)
        plt.scatter(outputs[i, :, 0], outputs[i, :, 1], 
                    c='red', label='Generated', s=5, alpha=0.6)
        plt.title(f'Cloud {i+1} Comparison')
        plt.legend()
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'training_out/{out_dir}/epoch_{epoch}_clouds.png')
    plt.close()


    # Plot loss curve separately
    plt.figure(figsize=(12, 6))
    window_size = 50
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(losses)), moving_avg, 
                color='orange')
    plt.plot(losses, alpha=0.3)
    plt.title('Training Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'training_out/{out_dir}/Loss_curve.png')
    plt.close()

def log_hyperparameters(args, out_dir, model_info=None):
    """
    Log all hyperparameters to a file for reproducibility.
    
    Args:
        args: The argparse Namespace containing hyperparameters
        out_dir: The output directory to save the log file
        model_info: Additional model information (optional)
    """
    hyperparams_file = os.path.join('training_out', out_dir, 'hyperparameters.txt')
    os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)
    
    with open(hyperparams_file, 'w') as f:
        f.write("==== Training Hyperparameters ====\n")
        # Get all attributes from args Namespace
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        
        # Add additional hyperparameters not specified in args
        f.write(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if model_info:
            f.write("\n==== Model Information ====\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")



def main():
    print("START")
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument("--filename", type=str, default="old_with_1D", help="Output directory name")
    parser.add_argument("--zdim", type=int, default=20, help="Latent dimension size")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--perturb_scale", type=float, default=0.0, help="Perturbation scale for latent functions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for nearest neighbor search")
    parser.add_argument("--pos_enc_L", type=int, default=7, help="Positional encoding parameter L")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_Z_samples", type=int, default=70, help="Number of latent function samples")
    parser.add_argument("--xdim", type=int, default=2, help="Input dimension")
    parser.add_argument("--num_points", type=int, default=6889, help="Number of points")
    parser.add_argument("--staleness", type=int, default=5, help="How often to update IMLE samples")
    parser.add_argument("--plot_epoch", type=int, default=250, help="How often to plot results")
    parser.add_argument("--batch_size", type=int, default=750, help="Batch size for training")
    parser.add_argument("--data_dir", type=str, default='/home/rsp8/scratch/Human_Point_Clouds/', help="Data directory path")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cosine", "step", "plateau"], 
                        help="Learning rate scheduler type")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the model")
    parser.add_argument("--model", type=str, default='new', help="Dropout rate for the model")
    
    args = parser.parse_args()
    num_points = int(int(np.sqrt(args.num_points))**2)
    output_dim = 3
    xdim = args.xdim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print system info
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model with efficient initialization
    if args.model == 'old':
        H_t = H_theta_Res_old_with_SE(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), 
                        output_dim=output_dim,
                        dropout_rate=args.dropout).to(device)
    else:
        H_t = H_theta_ResNet1D_Like_Old(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), 
                        output_dim=output_dim,
                        dropout_rate=args.dropout,
                        hidden_dims=[256, 512, 1024, 2048],
                        blocks_per_dim=[3, 4, 6, 3],
                        seblock = True).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in H_t.parameters())
    trainable_params = sum(p.numel() for p in H_t.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(H_t.parameters(), lr=args.lr, weight_decay = 1e-4)
    
    # Create scheduler if specified
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True)
    
    save_path = f'Out_{args.filename}/'
    
    # Log model info
    model_info = {
        "model_type": "H_theta_Res",
        "input_dim": args.zdim + int(args.pos_enc_L * 2 * args.xdim),
        "output_dim": output_dim,
        "optimizer": "AdamW",
        "weight_decay": 5e-4,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "scheduler": args.scheduler if args.scheduler else "None"
    }
    
    # Log hyperparameters before training
    log_hyperparameters(args, save_path, model_info)
    
    #Start training with timing
    print(f"Starting training with {args.epochs} epochs...")
    training_start = time.time()
    
    H_t, grad_norms, param_norms, losses = train(
        H_t,
        optimizer,
        out_dir=save_path,
        device=device,
        epochs=args.epochs,
        staleness=args.staleness,
        perturb_scale=args.perturb_scale,
        threshold=args.threshold,
        pos_enc_L=args.pos_enc_L,
        num_Z_samples=args.num_Z_samples,
        zdim=args.zdim,
        xdim=args.xdim,
        num_points=num_points,
        plot_epoch=args.plot_epoch,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        clip_grad_norm=args.clip_grad,
        lr_scheduler=scheduler
    )
    
    training_end = time.time()
    print(f"Training completed in {training_end - training_start:.2f} seconds")
    
    torch.save(H_t.state_dict(), f"training_out/Out_{args.filename}/H_t_weights.pth")
    num_points = 14400
    file_path = f"training_out/Out_{args.filename}/H_t_weights.pth"
    H_t.load_state_dict(torch.load(file_path))
    data_dir = '/home/rsp8/scratch/Human_Point_Clouds/'
    #data_dir = None
    if xdim == 1:
        x = torch.linspace(-0.05, 0.05, num_points).to(device).unsqueeze(1)
        data = (generate_data(num_points)).to(device)
    else:
        x1 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        x2 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        grid_x1, grid_x2 = torch.meshgrid((x1, x2), indexing='ij')
        data = (generate_3D_data(np.sqrt(num_points).astype(int))).to(device)[:,0:num_points,:]
        if data_dir is None:
            data = generate_3D_data(np.sqrt(6889).astype(int)).to(device)[:, 0:num_points, :]
            #data = torch.stack([data[i, torch.randperm(data.shape[1]), :] for i in range(data.shape[0])])
        else:
            data = []
            obj_files = sorted(glob.glob(os.path.join(data_dir, '*.obj')))
            #print(obj_files)
            for file_path in obj_files[4:34]:
                mesh = trimesh.load(file_path)
                vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                vertices = vertices - vertices.mean(0)
                data.append(vertices[0:num_points, :].unsqueeze(0))

            data = torch.cat(data, dim=0).to(device)
        x = torch.stack((grid_x1, grid_x2), dim=-1).reshape(-1, 2).to(device)
    z_in = pos_encoder_decaying(x, L=args.pos_enc_L).to(device)

    if output_dim == 2:
        plot_generated_curves_grid_2D(
            model=H_t,
            z_in=z_in,
            num_samps=40,
            data=data,
            out_dir=save_path,
            device=device,
            num_points=40,
            zdim=args.zdim,
            pos_enc_L=args.pos_enc_L,
            xdim=1,
            n_rows=10,
            n_cols=4,
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            figsize=(20, 50),
            save_dir=f'Generations_{args.filename}',
        )
    else:
        plot_interpolated_curves_3D(
            H_t=H_t,
            z_in=z_in,
            num_points=num_points,
            num_interpolations=50,
            device=device,
            zdim=args.zdim,
            pos_enc_L=args.pos_enc_L,
            save_dir=f'Generations_{args.filename}',
            data=data,
        )


if __name__ == "__main__":
    main()