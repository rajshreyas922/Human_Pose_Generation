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
        pos_enc_L=4,
        plot_epoch=2500,
        perturb_scale=0.97,
        threshold=0.0
):
    grad_norms = []
    param_norms = []
    losses = []
    os.makedirs('training_out/' + out_dir, exist_ok=True)

    if xdim == 1:
        x = torch.linspace(-0.05, 0.05, num_points).to(device).unsqueeze(1)
        data = (generate_data(num_points)).to(device)
    else:
        x1 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        x2 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        grid_x1, grid_x2 = torch.meshgrid((x1, x2), indexing='ij')
        data = (generate_3D_data(np.sqrt(num_points).astype(int))).to(device)[:,0:num_points,:]
        
        x = torch.stack((grid_x1, grid_x2), dim=-1).reshape(-1, 2).to(device)

    Zxs = torch.empty((num_Z_samples, num_points, zdim + int(pos_enc_L * 2 * xdim))).to(device)
    
    z_in = pos_encoder(x, L=pos_enc_L).to(device)
    for e in tqdm(range(epochs)):
        # Check if we need to update the stored model parameters
        if e % staleness == 0:
            Zs = generate_NN_latent_functions(num_samples=num_Z_samples, xdim=z_in.shape[1], zdim=zdim, bias=1)
            for i, model in enumerate(Zs):
                model = model.to(device)
                z = model(z_in)
                #z = F.normalize(z, p=2, dim=0)
                Zxs[i] = z.to(device)
            generated = H_t(Zxs).to(device)
            imle_nns = [find_nns(d, generated, threshold=threshold, disp=False) for d in data]
            imle_transformed_points = torch.empty((data.shape[0], num_points, zdim + int(pos_enc_L * 2 * xdim))).to(device)
            perturbed_Zs = []
            for i, (idx, _) in enumerate(imle_nns):
                model = Zs[idx]
                perturbed_model = copy.deepcopy(model)
                with torch.no_grad():
                    for param in perturbed_model.parameters():
                        param.add_(torch.randn_like(param) * perturb_scale)
                perturbed_Zs.append(perturbed_model)
                perturbed_model = perturbed_model.to(device)
                z = perturbed_model(z_in)
                imle_transformed_points[i] = z.to(device)

        # Zero gradients, calculate loss, backpropagate, and update weights
        optimizer.zero_grad()
        outs = H_t(imle_transformed_points)
        loss = f_loss(data, outs)
        losses.append(loss.item())
        if e % plot_epoch == 0 or e == epochs - 1:
            generated_disp = generated.to(device='cpu').detach().numpy()
            outs_disp = outs.to(device='cpu').detach().numpy()
            points_disp = data.to(device='cpu').detach().numpy()
            plt.figure(figsize=(15, 15))
            for i in range(data.shape[0]):
                line1 = plt.plot(outs_disp[i, :, 0], outs_disp[i, :, 1], marker='+')
                color = line1[0].get_color()
                plt.plot(points_disp[i, :, 0], points_disp[i, :, 1], marker='o', color=color)
            plt.title(f'Epoch: {e}')
            plt.savefig(f"training_out/{out_dir}/epoch_{e}.png")
            # Close plts
            plt.close()
            plt.figure(figsize=(15, 5))
            plt.plot(losses, label='Loss')
            plt.ylim(0, 15)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.savefig(f"training_out/{out_dir}/loss_curve.png")

        loss.backward()
        # grad_sum = 0
        # param_sum = 0
        # for param in H_t.parameters():
        #     param_sum += torch.norm(param) ** 2
        #     grad_sum += torch.norm(param.grad) ** 2

        # grad_norm = torch.sqrt(grad_sum).item()
        # param_norm = torch.sqrt(param_sum).item()
        # grad_norms.append(grad_norm)
        # param_norms.append(param_norm)
        optimizer.step()

    return H_t, grad_norms, param_norms, losses


def main():
    parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
    parser.add_argument("--filename", type=str, default="try_12", help="Output directory name")
    parser.add_argument("--zdim", type=int, default=10, help="Latent dimension size")
    parser.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument("--perturb_scale", type=float, default=0.4, help="Perturbation scale for latent functions")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for nearest neighbor search")
    parser.add_argument("--pos_enc_L", type=int, default=5, help="Positional encoding parameter L")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--num_Z_samples", type=int, default=70, help="Number of latent function samples")
    parser.add_argument("--xdim", type=int, default=2, help="Number of latent function samples")
    parser.add_argument("--num_points", type=int, default=484, help="Number of points")


    

    args = parser.parse_args()
    num_points = int(int(np.sqrt(args.num_points))**2)
    output_dim = 3
    xdim = args.xdim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_t = H_theta(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), output_dim=output_dim, num_layers=6, num_neurons=1500).to(device)
    #H_t = H_theta_skip(input_dim=args.zdim + int(args.pos_enc_L * 2 * args.xdim), output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(H_t.parameters(), lr=args.lr)
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
        num_points = num_points
    )
    
    torch.save(H_t.state_dict(), f"training_out/Out_{args.filename}/H_t_weights.pth")
    if xdim == 1:
        x = torch.linspace(-0.05, 0.05, num_points).to(device).unsqueeze(1)
        data = (generate_data(num_points)).to(device)
    else:
        x1 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        x2 = torch.linspace(-0.05, 0.05, int(np.sqrt(num_points)))
        grid_x1, grid_x2 = torch.meshgrid((x1, x2), indexing='ij')
        data = (generate_3D_data(np.sqrt(num_points).astype(int))).to(device)[:,0:num_points,:]
        x = torch.stack((grid_x1, grid_x2), dim=-1).reshape(-1, 2).to(device)
    z_in = pos_encoder(x, L=args.pos_enc_L).to(device)
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
            num_samples=40,
            device=device,
            zdim=args.zdim,
            pos_enc_L=args.pos_enc_L,
            save_dir=f'Generations_{args.filename}',
            data=data,

        )


if __name__ == "__main__":
    main()