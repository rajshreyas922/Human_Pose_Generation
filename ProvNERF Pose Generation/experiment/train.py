import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from models import *
from generate_data import *

from sklearn.model_selection import ParameterGrid  # Added import

def generate_NN_latent_functions(num_samples, xdim=1, zdim=2, bias=0):
    class NN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 500)
            self.fc3 = nn.Linear(500, 500)
            self.fc4 = nn.Linear(500, output_dim)

            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            with torch.no_grad():
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
            return x

    def weights_init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=bias)

    networks = []
    for _ in range(num_samples):
        net = NN(xdim, zdim)
        net.apply(weights_init_normal)
        networks.append(net)
    return networks

def pos_encoder(x, L):
    _, n = x.shape
    encoding = []
    for i in range(n):
        for l in range(L):
            encoding.append(torch.sin(1.1*(2**l) * torch.pi * x[:, i:i+1]))
            encoding.append(torch.cos(1.1*(2**l) * torch.pi * x[:, i:i+1]))
    encoded_x = torch.cat(encoding, dim=-1)*5
    return encoded_x

def find_nns(Y, G, disp=False):
    distances = torch.sum(((Y - G) ** 2), dim=2).mean(dim=1)
    min_, min_idx = torch.min(distances, dim=0)
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

def train_model(
    H_t, 
    optimizer, 
    scheduler, 
    epochs, 
    staleness, 
    num_Z_samples, 
    num_points, 
    zdim, 
    z_scale, 
    plot_epoch, 
    pos_enc_L, 
    device,
    param_name,
    data):

    losses = []
    grad_norms = []
    param_norms = []
    Zxs = torch.empty((num_Z_samples, num_points, zdim)).to(device)
    z_in = pos_encoder(torch.linspace(-0.05, 0.05, num_points).to(device).unsqueeze(1), L=pos_enc_L)

    for e in tqdm(range(epochs), desc=f"Training {param_name}"):
        if e % staleness == 0:
            Zs = generate_NN_latent_functions(num_samples=num_Z_samples, xdim=z_in.shape[1], zdim=zdim, bias=1)
            for i, model in enumerate(Zs):
                model = model.to(device)
                Zxs[i] = (model(z_in) / z_scale).to(device)
            generated = H_t(Zxs).to(device)
            imle_nns = [find_nns(d, generated) for d in data]
            imle_transformed_points = torch.empty((data.shape[0], num_points, zdim)).to(device)

            perturbed_Zs = []
            for i, idx in enumerate(imle_nns):
                model = Zs[idx]
                perturbed_model = copy.deepcopy(model)
                with torch.no_grad():
                    for param in perturbed_model.parameters():
                        param.add_(torch.randn_like(param) * 0.2)
                perturbed_Zs.append(perturbed_model)
                perturbed_model = perturbed_model.to(device)
                imle_transformed_points[i] = (perturbed_model(z_in) / z_scale).to(device)

        optimizer.zero_grad()
        outs = H_t(imle_transformed_points)
        loss = f_loss(data, outs)
        losses.append(loss.item())

        if e % plot_epoch == 0 or e == epochs - 1:
            generated_disp = generated.to('cpu').detach().numpy()
            outs_disp = outs.to('cpu').detach().numpy()
            points_disp = data.to('cpu').detach().numpy()
            plt.figure(figsize=(15, 15))
            for i in range(data.shape[0]):
                line1 = plt.plot(outs_disp[i, :, 0], outs_disp[i, :, 1], marker='+')
                color = line1[0].get_color()
                plt.plot(points_disp[i, :, 0], points_disp[i, :, 1], marker='o', color=color)
            plt.title(f'Epoch: {e}')
            plt.savefig(f"plots/{param_name}/epoch_{e}.png")
            plt.close()

            plt.figure(figsize=(15, 5))
            plt.plot(losses, label='Loss')
            plt.ylim(0, 15)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.show()

        loss.backward()
        grad_sum = 0
        param_sum = 0
        for param in H_t.parameters():
            param_sum += torch.norm(param) ** 2
            grad_sum += torch.norm(param.grad) ** 2

        grad_norm = torch.sqrt(grad_sum).item()
        param_norm = torch.sqrt(param_sum).item()
        grad_norms.append(grad_norm)
        param_norms.append(param_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    plt.figure(figsize=(15, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f'plots/{param_name}/Loss Curve.png')
    plt.show()

    return H_t


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_grid = {
        #Training
        "epochs": [1000, 2000, 3000, 4000, 5000],
        "staleness": [2, 5, 10, 50, 100],
        "lr": [0.01, 0.005, 0.0001, 1e-4, 5e-4, 8e-4, 5e-5, 1e-5],
        "b1": [0.85, 0.9, 0.95, 0.99],
        "b2": [0.99, 0.999, 0.995],
        "zdim": [5, 6, 7, 8],
        "pos_enc_L": [2, 5, 6, 7, 8],
        "scheduler": [True, False],
        "num_Z_samples": [30, 45, 150, 300],

        #If scheduler:
        "gamma": [0.95, 0.9, 0.85],
        "step": [50, 100, 1000],
        "z_scale": [100, 500, 1000],

        #Model
        "architecture": ['regular, inject'],
        "depth": [4, 5, 6],
        "width": [500, 1000, 2000, 2500],

        #If inject:
        "injection_depth": [4, 5, 6],
        "injection_width": [500, 1000, 2000, 2500],
        "one_vec": [5, 20, 30, 100],

        #Data
        "num_points": [20, 50, 100],
        "t_range": [[-0.01, 0.01], [-0.05, 0.05], [-0.01, 0.01]],
        "num_curves": [2, 5, 20, 50]
    }

    print("Params")

    # Use ParameterGrid instead of itertools.product
    param_combinations = ParameterGrid(param_grid)

    torch.manual_seed(1)
    np.random.seed(1)

    with tqdm(total=len(param_combinations), desc="Parameter Grid Search") as pbar:
        for idx, params in enumerate(param_combinations):
            plot_epoch = params["epochs"] // 10
            param_name = f"run_{idx + 1}"
            os.makedirs(f'plots/{param_name}', exist_ok=True)
            os.makedirs(f'plots/{param_name}/outputs/', exist_ok=True)

            data = generate_data(num_points=params["num_points"]).to(device)
            if params["architecture"] == 'regular':
                H_t = H_theta_new(
                    input_dim=params["zdim"],
                    output_dim=2,  # Assuming fixed output dimension
                    num_layers=params["depth"],
                    num_neurons=params["width"],
                    num_layers_inject=params["injection_depth"],
                    num_neuron_inject=params["injection_width"]
                ).to(device)
            else:
                H_t = H_theta(
                    input_dim=params["zdim"],
                    output_dim=2,  # Assuming fixed output dimension
                    num_layers=params["depth"],
                    num_neurons=params["width"],
                    num_layers_inject=params["injection_depth"],
                    num_neuron_inject=params["injection_width"]
                ).to(device)

            optimizer = optim.AdamW(H_t.parameters(), lr=params["lr"], betas=(params["b1"], params["b2"]), eps=1e-8)

            scheduler = None
            if params["scheduler"]:
                scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=params["gamma"], step_size=params["step"])

            model = train_model(
                H_t=H_t,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=params["epochs"],
                staleness=params["staleness"],
                num_Z_samples=params["num_Z_samples"],  
                num_points=params["num_points"],
                zdim=params["zdim"],
                z_scale=params["z_scale"],  
                plot_epoch=plot_epoch,  
                pos_enc_L=params["pos_enc_L"],
                device=device,
                param_name=param_name,
                data=data
            )

            eval(model, data, num_samples=50, ouput_dir=f"plots/{param_name}/outputs/")

            pbar.update(1)
            print(f"Completed training for parameter set {idx + 1}/{len(param_combinations)}")
