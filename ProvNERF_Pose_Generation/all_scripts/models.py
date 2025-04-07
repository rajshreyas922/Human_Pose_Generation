import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetRes(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=512, depth=3):
        super().__init__()
        self.depth = depth
        
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Encoder components
        self.encoder_blocks = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        current_dim = hidden_dim
        for _ in range(depth):
            # Create residual blocks and scales for this stage
            res_blocks = nn.ModuleList()
            scales = nn.ParameterList()
            for _ in range(2):  # 2 residual blocks per stage
                res_blocks.append(nn.Sequential(
                    nn.Linear(current_dim, current_dim),
                    nn.LayerNorm(current_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(current_dim, current_dim),
                    nn.LayerNorm(current_dim),
                ))
                scales.append(nn.Parameter(torch.tensor(0.5)))
            
            # Create a container module for this encoder stage
            stage = nn.ModuleDict({
                'blocks': res_blocks,
                'scales': scales
            })
            self.encoder_blocks.append(stage)
            
            # Downsampling
            self.down_layers.append(nn.Sequential(
                nn.Linear(current_dim, current_dim//2),
                nn.LayerNorm(current_dim//2),
                nn.LeakyReLU(0.2)
            ))
            current_dim = current_dim // 2

        # Bottleneck
        self.bottleneck_blocks = nn.ModuleList()
        self.bottleneck_scales = nn.ParameterList()
        for _ in range(2):
            self.bottleneck_blocks.append(nn.Sequential(
                nn.Linear(current_dim, current_dim),
                nn.LayerNorm(current_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(current_dim, current_dim),
                nn.LayerNorm(current_dim),
            ))
            self.bottleneck_scales.append(nn.Parameter(torch.tensor(0.5)))

        # Decoder components
        self.decoder_blocks = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        for _ in range(depth):
            # Upsampling
            self.up_layers.append(nn.Sequential(
                nn.Linear(current_dim, current_dim*2),
                nn.LayerNorm(current_dim*2),
                nn.LeakyReLU(0.2)
            ))
            current_dim *= 2
            
            # Create residual blocks and scales for this stage
            res_blocks = nn.ModuleList()
            scales = nn.ParameterList()
            for _ in range(2):  # 2 residual blocks per stage
                res_blocks.append(nn.Sequential(
                    nn.Linear(current_dim, current_dim),
                    nn.LayerNorm(current_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(current_dim, current_dim),
                    nn.LayerNorm(current_dim),
                ))
                scales.append(nn.Parameter(torch.tensor(0.5)))
            
            # Create a container module for this decoder stage
            stage = nn.ModuleDict({
                'blocks': res_blocks,
                'scales': scales
            })
            self.decoder_blocks.append(stage)

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        skips = []
        
        # Encoder path
        x = self.input_layer(x)
        for i in range(self.depth):
            # Process residual blocks
            stage = self.encoder_blocks[i]
            for j, block in enumerate(stage['blocks']):
                residual = x
                x = block(x)
                x = x * stage['scales'][j] + residual
                x = F.leaky_relu(x, 0.2)
            skips.append(x)
            
            # Downsample
            x = self.down_layers[i](x)

        # Bottleneck
        for j, block in enumerate(self.bottleneck_blocks):
            residual = x
            x = block(x)
            x = x * self.bottleneck_scales[j] + residual
            x = F.leaky_relu(x, 0.2)

        # Decoder path
        for i in range(self.depth):
            # Upsample
            x = self.up_layers[i](x)
            
            # Add skip connection
            x = x + skips.pop()
            
            # Process residual blocks
            stage = self.decoder_blocks[i]
            for j, block in enumerate(stage['blocks']):
                residual = x
                x = block(x)
                x = x * stage['scales'][j] + residual
                x = F.leaky_relu(x, 0.2)

        return self.output_layer(x)


class H_theta_Res(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=512):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2) 
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),  
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(12)  
        ])

        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(len(self.res_blocks))  # Scale factors for residual connections
        ])
        
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)

        for i, block in enumerate(self.res_blocks):
            residual = x
            x = block(x)
            x = x * self.res_scales[i] + residual
            x = F.leaky_relu(x, 0.2)  
        
        return self.final_layer(x)
        
class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=5, num_neurons=2000):
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


class H_theta_skip(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output.
        """
        super(H_theta_skip, self).__init__()
        self.input_dim = input_dim
        
        # Define six layers with skip connections every 2 layers.
        self.layer1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU()
        
        self.layer5 = nn.Linear(2000, 2000)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(2000, 2000)
        self.relu6 = nn.ReLU()
        
        # Output layer: maps processed features to the desired output dimension.
        self.output_layer = nn.Linear(2000, output_dim)

    def forward(self, x, disp=False):
        """
        Forward pass through the network with skip connections every 2 layers.
        
        Args:
            x (Tensor): Input tensor of shape [batch, sequence, features] (or similar).
            disp (bool): Optional flag for display/debug (currently unused).
        
        Returns:
            Tensor: Output of the network.
        """
        # Save original input for the first skip connection
        input_skip = x
        
        # First two layers
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        
        # Add skip connection after the first two layers
        if x.shape == input_skip.shape:
            x = x + input_skip  # Skip connection by addition
        
        # Next two layers
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        
        # Add skip connection after the next two layers
        if x.shape == input_skip.shape:
            x = x + input_skip  # Skip connection by addition
        
        # Final two layers
        x = self.relu5(self.layer5(x))
        x = self.relu6(self.layer6(x))
        
        # Add skip connection after the final two layers
        if x.shape == input_skip.shape:
            x = x + input_skip  # Skip connection by addition
        
        # Final output layer
        out = self.output_layer(x)
        return out


# A neural network that uses Adaptive Instance Normalization (AdaIN) style injection
class H_theta_AdaIN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=10, num_layers_inject=4, num_neuron_inject=500, num_neurons=50):
        """
        Args:
            input_dim (int): Dimensionality of the conditioning/injection input.
            output_dim (int): Dimensionality of the final output.
            num_layers (int): Total number of layers for the base network (must be even).
            num_layers_inject (int): Number of layers for the injection network.
            num_neuron_inject (int): Number of neurons per injection layer.
            num_neurons (int): Number of neurons in each layer of the base network.
        """
        super(H_theta_AdaIN, self).__init__()
        
        # Ensure that the number of layers is even to split equally into before and after injection.
        assert num_layers % 2 == 0
        half_layers = num_layers // 2

        # Build the part of the network before the AdaIN injection.
        layers_before_inject = []
        for i in range(half_layers):
            if i == 0:
                # The input to this network is a tensor of ones with 5 features.
                layers_before_inject.append(nn.Linear(5, num_neurons))
            else:
                layers_before_inject.append(nn.Linear(num_neurons, num_neurons))
            layers_before_inject.append(nn.ReLU())
        self.model_before_inject = nn.Sequential(*layers_before_inject)

        # Build the injection network that computes scale and shift parameters.
        inject_layers = []
        for i in range(num_layers_inject):
            if i == 0:
                # First injection layer maps from the conditioning input dimension to num_neuron_inject.
                inject_layers.append(nn.Linear(input_dim, num_neuron_inject))
            else:
                inject_layers.append(nn.Linear(num_neuron_inject, num_neuron_inject))
            inject_layers.append(nn.ReLU())
        # Final injection layer outputs 2 values: one for scaling and one for shifting.
        inject_layers.append(nn.Linear(num_neuron_inject, 2))
        self.inject = nn.Sequential(*inject_layers)

        # Build the part of the network after AdaIN injection.
        layers_after_inject = []
        for i in range(half_layers):
            layers_after_inject.append(nn.Linear(num_neurons, num_neurons))
            layers_after_inject.append(nn.ReLU())
        # Final output layer
        layers_after_inject.append(nn.Linear(num_neurons, output_dim))
        self.model_after_inject = nn.Sequential(*layers_after_inject)

        # Layer normalization for stabilizing the features before AdaIN is applied.
        self.layer_norm = nn.LayerNorm(normalized_shape=num_neurons)

    def forward(self, x, disp=False):
        """
        Forward pass through the AdaIN-based network.
        
        Args:
            x (Tensor): Conditioning input tensor.
            disp (bool): Optional flag for display/debug (currently unused).
        
        Returns:
            Tensor: Output of the network.
        """
        # Create a constant input of ones with shape compatible with the base network input.
        # Assuming the shape of x is [batch, sequence, input_dim], we use 5 features as specified.
        y = self.model_before_inject(torch.ones((x.shape[0], x.shape[1], 5), device=x.device))
        
        # Pass the conditioning input through the injection network to get scaling and shifting factors.
        sc = self.inject(x)
        
        # Calculate scale (s) ensuring it is at least 1 (after adding 1 and applying ReLU on the offset).
        s = torch.max(torch.tensor(0., device=x.device), 1 + sc[:, :, 0].unsqueeze(-1))
        
        # Get the shift (c) parameter.
        c = sc[:, :, 1].unsqueeze(-1)
        
        # Normalize the base network features.
        y = self.layer_norm(y)
        
        # Apply the AdaIN operation: scale and shift the normalized features.
        x = (s * y + c).to(x.device)
        
        # Process the adapted features through the remaining part of the network.
        x = self.model_after_inject(x).to(x.device)
        return x
