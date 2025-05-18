import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os
import argparse

from generate import plot_generated_curves_3D, plot_generated_curves_grid_2D
from imle import generate_NN_latent_functions, find_nns, f_loss
from models import *
from data_process import *

from misc import *
import trimesh
import glob
import time

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
        data_dir=None
):
    grad_norms = []
    param_norms = []
    losses = []
    os.makedirs('training_out/' + out_dir, exist_ok=True)
    batch_size = 500
    indices = torch.randperm(num_points)
    batches = torch.split(indices, batch_size)  # Automatically handles remainders
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2500, gamma = 0.3)
    #gamma = 0.3 ** (1.0 / 2500)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)  
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #optimizer, T_max=7000, eta_min=1e-5)
    # Data generation
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
            #print(obj_files)
            for file_path in obj_files[4:14]:
                mesh = trimesh.load(file_path)
                vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                vertices = vertices - vertices.mean(0)
                data.append(vertices[0:num_points, :].unsqueeze(0))

            data = torch.cat(data, dim=0).to(device)
    #print(data.shape)
    z_in = pos_encoder_decaying(x, L=pos_enc_L).to(device)
 
    stored_imle_transformed_points = []
    total_time_z_generation = 0.0
    total_time_perturbation = 0.0
    total_time_reuse = 0.0

    for e in tqdm(range(epochs)):
        epoch_loss = 0
        # if e % 2500 == 0:
        #     perturb_scale *= 1.1
        if e % staleness == 0:
            stored_imle_transformed_points = []

        for batch_idx, batch_indices in enumerate(batches):
            current_batch_size = len(batch_indices)
            data_batch = data[:, batch_indices, :]
            #print(data_batch.shape)
            if e % staleness == 0:
                z_gen_start = time.time()

                Zs = generate_NN_latent_functions(
                    num_samples=num_Z_samples,
                    xdim=z_in.shape[1],
                    zdim=zdim,
                    bias=1
                )

                Zxs = torch.empty((num_Z_samples, current_batch_size, zdim + int(pos_enc_L * 2 * xdim))).to(device)
                
                for i, model in enumerate(Zs):
                    model = model.to(device)
                    with torch.no_grad():
                        z = model(z_in[batch_indices])
                    Zxs[i] = z

                with torch.no_grad():
                    generated = H_t(Zxs)
                    imle_nns = [find_nns(d, generated, threshold=threshold) for d in data_batch]

                z_gen_end = time.time()
                total_time_z_generation += z_gen_end - z_gen_start

                perturb_start = time.time()

                imle_transformed_points = torch.empty((data.shape[0], current_batch_size, zdim + int(pos_enc_L * 2 * xdim))).to(device)

                for i, (nn_idx, _) in enumerate(imle_nns):
                    original_model = Zs[nn_idx]
                    perturbed_model = copy.deepcopy(original_model).to(device)
                    with torch.no_grad():
                        for param in perturbed_model.parameters():
                            param += torch.randn_like(param) * perturb_scale
                        z_perturbed = perturbed_model(z_in[batch_indices])
                    imle_transformed_points[i] = z_perturbed

                perturb_end = time.time()
                total_time_perturbation += perturb_end - perturb_start

                stored_imle_transformed_points.append(imle_transformed_points)
            else:
                reuse_start = time.time()
                imle_transformed_points = stored_imle_transformed_points[batch_idx]
                reuse_end = time.time()
                total_time_reuse += reuse_end - reuse_start

            
            optimizer.zero_grad()
            outputs = H_t(imle_transformed_points)
            loss = f_loss(data_batch, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #scheduler.step()
        avg_loss = np.log10(epoch_loss)
        losses.append(avg_loss)


        if e % plot_epoch == 0 or e == epochs-1:
            plot_results(
                data.cpu().numpy(),
                outputs.detach().cpu().numpy(),
                losses,
                e,
                epochs,
                out_dir
            )

    print("\n==== Timing Summary ====")
    print(f"Total time spent generating latent functions (Zs): {total_time_z_generation:.4f} s")
    print(f"Total time spent perturbing and selecting IMLE points: {total_time_perturbation:.4f} s")
    print(f"Total time spent reusing stored IMLE points: {total_time_reuse:.4f} s")


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



def main():
    print("START")
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument("--filename", type=str, default="profiling_pe_8_decay_0.5_10_poses_zdim_10_1024_8_model_500_batch_size", help="Output directory name")
    parser.add_argument("--zdim", type=int, default=10, help="Latent dimension size")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--perturb_scale", type=float, default=0.0, help="Perturbation scale for latent functions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for nearest neighbor search")
    parser.add_argument("--pos_enc_L", type=int, default=8, help="Positional encoding parameter L")
    parser.add_argument("--lr", type=float, default=(1e-4)/5, help="Learning rate for the optimizer")
    parser.add_argument("--num_Z_samples", type=int, default=100, help="Number of latent function samples")
    parser.add_argument("--xdim", type=int, default=2, help="Number of latent function samples")
    parser.add_argument("--num_points", type=int, default=6889, help="Number of points")

    

    args = parser.parse_args()
    num_points = int(int(np.sqrt(args.num_points))**2)
    output_dim = 3
    xdim = args.xdim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_t = H_theta_Res(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), output_dim=output_dim).to(device)
    #H_t = H_theta(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), output_dim=output_dim).to(device)
    #H_t = ResnetRS()
    
    #H_t = H_theta_skip(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(H_t.parameters(), lr=args.lr, weight_decay = 5e-4)
    save_path = f'Out_{args.filename}/'
    H_t, grad_norms, param_norms, losses = train(
        H_t,
        optimizer,
        out_dir=save_path,
        device=device,
        epochs=args.epochs,
        perturb_scale=args.perturb_scale,
        threshold=args.threshold,
        pos_enc_L=args.pos_enc_L,
        num_Z_samples=args.num_Z_samples,
        zdim=args.zdim,
        xdim=args.xdim,
        num_points = num_points,
        data_dir='/home/rsp8/scratch/Human_Point_Clouds/'
    )
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
            for file_path in obj_files[4:14]:
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
        plot_generated_curves_3D(
            H_t=H_t,
            z_in=z_in,
            num_points=num_points,
            num_samples=20,
            device=device,
            zdim=args.zdim,
            pos_enc_L=args.pos_enc_L,
            save_dir=f'Generations_{args.filename}',
            data=data,

        )


if __name__ == "__main__":
    main()