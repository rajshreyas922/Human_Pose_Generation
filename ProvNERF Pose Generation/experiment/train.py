import torch
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from models import *
from generate_data import *
import csv
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
            outs_disp = outs.to('cpu').detach().numpy()
            points_disp = data.to('cpu').detach().numpy()
            plt.figure(figsize=(15, 15))
            for i in range(data.shape[0]):
                line1 = plt.plot(outs_disp[i, :, 0], outs_disp[i, :, 1], marker='+')
                color = line1[0].get_color()
                plt.plot(points_disp[i, :, 0], points_disp[i, :, 1], marker='o', color=color)
            plt.title(f'Epoch: {e}')
            plt.savefig(f"correct_plots/{param_name}/epoch_{e}.png")
            plt.close()

            # plt.figure(figsize=(15, 5))
            # plt.plot(losses, label='Loss')
            # plt.ylim(0, 15)
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.title('Loss Curve')
            # plt.legend()

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


    plt.figure(figsize=(15, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f'correct_plots/{param_name}/Loss Curve.png')


    return H_t


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Common parameters for all runs (excluding architecture/depth/width/injection values).
    common_params = {
        # Training
        "epochs": 5000,             # <-- updated to 5500
        "staleness": 5,
        "lr": 5e-5,
        "b1": 0.85,
        "b2": 0.999,
        "zdim": 30,
        "pos_enc_L": 6,
        "scheduler": False,
        "num_Z_samples": 100,

        # If scheduler:
        "gamma": 0.9,
        "step": 1000,
        "z_scale": 100,

        # Data
        "num_points": 20,
        "t_range": [-0.05, 0.05],
        "num_curves": 20,

        # This is used only if injection is needed
        "one_vec": 5,
    }

    # Manually specify the model architectures we want to test:
    architecture_combos = [
        # Regular
        {"architecture": "regular", "depth": 4,  "width": 2000, "injection_depth": None, "injection_width": None},

        {"architecture": "regular", "depth": 6, "width": 2000,  "injection_depth": None, "injection_width": None},
        {"architecture": "regular", "depth": 4, "width": 1000,  "injection_depth": None, "injection_width": None},

# ========================================================================================================================

        # Inject
        {"architecture": "inject",  "depth": 4,  "width": 2000, "injection_depth": 3,  "injection_width": 1000},

        {"architecture": "inject",  "depth": 6, "width": 2000,  "injection_depth": 3,  "injection_width": 1000},
        {"architecture": "inject",  "depth": 4, "width": 1000,  "injection_depth": 3,  "injection_width": 1000},

# ========================================================================================================================

        {"architecture": "inject",  "depth": 4,  "width": 2000, "injection_depth": 5,  "injection_width": 1000},

        {"architecture": "inject",  "depth": 6, "width": 2000,  "injection_depth": 5,  "injection_width": 1000},
        {"architecture": "inject",  "depth": 4, "width": 1000,  "injection_depth": 5,  "injection_width": 1000},
    ]

    # Build the final list of parameter combos by merging common_params with each architecture combo
    param_combinations = []
    for arch_conf in architecture_combos:
        # Make a copy of common_params for each combo
        combo = dict(common_params)
        # Update the copy with our architecture/dim specifics
        combo.update(arch_conf)
        param_combinations.append(combo)

    # CSV header
    csv_file_path = "parameter_combinations_during_training_correct.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Index"] + list(param_combinations[0].keys()))
        writer.writeheader()

    torch.manual_seed(1)
    np.random.seed(1)

    with tqdm(total=len(param_combinations), desc="Parameter Grid Search") as pbar:
        for idx, params in enumerate(param_combinations, start=1):
            plot_epoch = params["epochs"] // 10
            param_name = f"run_{idx}"
            os.makedirs(f'correct_plots/{param_name}', exist_ok=True)

            # Save current parameter combination to CSV
            params_with_index = {"Index": idx}
            params_with_index.update(params)
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["Index"] + list(params.keys()))
                writer.writerow(params_with_index)

            # Generate data
            data = generate_data(num_points=params["num_points"], num_curves=params["num_curves"]).to(device)

            # Construct the model
            if params["architecture"] == 'regular':
                # injection_* parameters won't matter here
                H_t = H_theta_new(
                    input_dim=params["zdim"],
                    output_dim=2,  # fixed output dimension
                    num_layers=params["depth"],
                    num_neurons=params["width"],
                    # Provide valid injection placeholders (they won't be used)
                    num_layers_inject=2,
                    num_neuron_inject=500
                ).to(device)
            else:
                # 'inject' architecture
                H_t = H_theta(
                    input_dim=params["zdim"],
                    output_dim=2,  # fixed output dimension
                    num_layers=params["depth"],
                    num_neurons=params["width"],
                ).to(device)

            print(H_t)

            optimizer = optim.AdamW(
                H_t.parameters(),
                lr=params["lr"],
                betas=(params["b1"], params["b2"]),
                eps=1e-8
            )

            # Optional scheduler
            scheduler = None
            if params["scheduler"]:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    gamma=params["gamma"],
                    step_size=params["step"]
                )

            # Train the model
            trained_model = train_model(
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

            pbar.update(1)
            print(f"Completed training for parameter set {idx}/{len(param_combinations)}")
