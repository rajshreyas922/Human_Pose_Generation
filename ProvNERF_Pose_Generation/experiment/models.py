import torch
import torch.nn as nn

class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_neurons):
        super(H_theta, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, num_neurons))
            else:
                layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_neurons, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, disp=False):
        out = self.model(x)
        return out


class H_theta_new(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers = 10, num_layers_inject = 4, num_neuron_inject = 500, num_neurons = 50):
        super(H_theta_new, self).__init__()
        assert num_layers % 2 == 0, "Number of layers must be even for injection to occur after half."

        half_layers = num_layers // 2

        # First half of the model before injection
        layers_before_inject = []
        for i in range(half_layers):
            if i == 0:
                layers_before_inject.append(nn.Linear(5, num_neurons))
            else:
                layers_before_inject.append(nn.Linear(num_neurons, num_neurons))
            layers_before_inject.append(nn.ReLU())
        self.model_before_inject = nn.Sequential(*layers_before_inject)

        # Injection submodel
        inject_layers = []
        for i in range(num_layers_inject):
            if i == 0:
                inject_layers.append(nn.Linear(input_dim, num_neuron_inject))
            else:
                inject_layers.append(nn.Linear(num_neuron_inject, num_neuron_inject))
            inject_layers.append(nn.ReLU())
        inject_layers.append(nn.Linear(num_neuron_inject, 2))
        self.inject = nn.Sequential(*inject_layers)

        # Second half of the model after injection
        layers_after_inject = []
        for i in range(half_layers):
            layers_after_inject.append(nn.Linear(num_neurons, num_neurons))
            layers_after_inject.append(nn.ReLU())
        layers_after_inject.append(nn.Linear(num_neurons, output_dim))
        self.model_after_inject = nn.Sequential(*layers_after_inject)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=num_neurons)

    def forward(self, x, disp=False):
        # First half
        y = self.model_before_inject(torch.ones((x.shape[0], x.shape[1], 5), device=x.device))

        # Injection
        sc = self.inject(x)
        s = torch.max(torch.tensor(0., device=x.device), 1 + sc[:, :, 0].unsqueeze(-1))
        c = sc[:, :, 1].unsqueeze(-1)

        # Apply normalization and combine
        y = self.layer_norm(y)
        x = (s * y + c).to(x.device)

        # Second half
        x = self.model_after_inject(x).to(x.device)

        if disp:
            print(f"Input shape: {x.shape}, Injection shape: {sc.shape}, Output shape: {x.shape}")

        return x