import torch
import torch.nn as nn
import torch.nn.functional as F

# A fully connected neural network with many layers.
class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=70, num_neurons=20):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output.
            num_layers (int): Total number of linear layers (excluding final layer).
            num_neurons (int): Number of neurons in each hidden layer.
        """
        super(H_theta, self).__init__()
        layers = []  # List to store layers sequentially

        # Create a stack of linear layers with ReLU activation
        for i in range(num_layers):
            if i == 0:
                # First layer maps input_dim -> num_neurons
                layers.append(nn.Linear(input_dim, num_neurons))
            else:
                # Subsequent layers: num_neurons -> num_neurons
                layers.append(nn.Linear(num_neurons, num_neurons))
            # Add ReLU activation after each linear layer
            layers.append(nn.ReLU())
        
        # Final output layer maps from num_neurons to output_dim
        layers.append(nn.Linear(num_neurons, output_dim))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, disp=False):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
            disp (bool): Optional flag for display/debug (currently unused).
        
        Returns:
            Tensor: Output of the network.
        """
        out = self.model(x)
        return out


# A neural network with a skip connection that concatenates the input
class H_theta_skip(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            output_dim (int): Dimensionality of output.
        """
        super(H_theta_skip, self).__init__()
        self.input_dim = input_dim
        
        # First three layers: expand and process the input with 2000 neurons each.
        self.layer1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU()
        
        # Fourth layer: takes concatenated features (skip connection) as input.
        # Concatenation is along dimension 2, so the input size becomes 2000 + input_dim.
        self.layer4 = nn.Linear(2000 + input_dim, 2000)
        self.relu4 = nn.ReLU()
        
        # Output layer: maps processed features to the desired output dimension.
        self.output_layer = nn.Linear(2000, output_dim)

    def forward(self, x, disp=False):
        """
        Forward pass through the network with skip connection.
        
        Args:
            x (Tensor): Input tensor of shape [batch, sequence, features] (or similar).
            disp (bool): Optional flag for display/debug (currently unused).
        
        Returns:
            Tensor: Output of the network.
        """
        # Save original input for skip connection
        input_skip = x
        
        # Process input through first three layers
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        
        # Concatenate the skip connection along the feature dimension (assumed dim=2)
        x = torch.cat((x, input_skip), dim=2)
        
        # Process concatenated tensor through the next layer
        x = self.relu4(self.layer4(x))
        
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
