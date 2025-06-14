import torch
import torch.nn as nn
import torch.nn.functional as F

class H_theta_Res_old(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=1024, num_blocks = 8, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Add dropout after activation
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),  # Add dropout within the block
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(num_blocks) 
        ])

        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(len(self.res_blocks))
        ])
        
        self.dropout = nn.Dropout(dropout_rate)  # Additional dropout before final layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)

        for i, block in enumerate(self.res_blocks):
            residual = x
            x = block(x)
            x = x * self.res_scales[i] + residual
            x = F.leaky_relu(x, 0.2)
        
        x = self.dropout(x)  # Apply dropout before final projection
        return self.final_layer(x)
    

class SEBlock1D(nn.Module):
    def __init__(self, hidden_dim, reduction_ratio=0.25):
        super().__init__()
        reduced_dim = int(hidden_dim * reduction_ratio)
        self.squeeze_excite = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [num_curves, num_points, hidden_dim]
        orig = x
    
        # [num_curves, num_points, hidden_dim] -> [num_curves, hidden_dim]
        x = x.mean(dim=1)
        x = self.squeeze_excite(x)  # [num_curves, hidden_dim]
        x = x.unsqueeze(1)
        
        return orig * x


class H_theta_Res_old_with_SE(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=1024, num_blocks=8, 
                 dropout_rate=0.2, seblock=True, reduction_ratio=0.25):
        super().__init__()
        self.seblock_enabled = seblock
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(num_blocks) 
        ])
        
        # Add SE blocks for each residual block
        if self.seblock_enabled:
            self.se_blocks = nn.ModuleList([
                SEBlock1D(hidden_dim, reduction_ratio) for _ in range(num_blocks)
            ])

        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(len(self.res_blocks))
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [num_curves, num_points, input_dim]
        x = self.input_layer(x)  # [num_curves, num_points, hidden_dim]

        for i, block in enumerate(self.res_blocks):
            residual = x
            x = block(x)
            
            # Apply SE block if enabled
            if self.seblock_enabled:
                x = self.se_blocks[i](x)
            
            x = x * self.res_scales[i] + residual
            x = F.leaky_relu(x, 0.2)
        
        x = self.dropout(x)
        return self.final_layer(x)


   
class SEBlock1D_Conv(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [num_curves, channels, num_points]
        se_weights = self.squeeze_excite(x) 
        return x * se_weights

class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, 
                 stride=1, downsample=None, seblock=True, reduction_ratio=0.25,
                 dropout_rate=0.1):
        super().__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        
        # 1x1 conv
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=1, 
                              stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # SE block
        self.seblock = SEBlock1D_Conv(out_channels, reduction_ratio) if seblock else nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply SE block
        out = self.seblock(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class H_theta_ResNet1D(nn.Module):
    def __init__(self, input_channels, output_dim=3, dropout_rate=0.2, 
                 seblock=True, reduction_ratio=0.25):
        super().__init__()
        
        # Stem block
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Build ResNet layers following the table structure
        self.layer1 = self._make_layer(64, 64, 256, 3, seblock, reduction_ratio, dropout_rate)
        self.layer2 = self._make_layer(256, 128, 512, 4, seblock, reduction_ratio, dropout_rate, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, 6, seblock, reduction_ratio, dropout_rate, stride=2)  # Using 6 instead of 23 for computational efficiency
        self.layer4 = self._make_layer(1024, 512, 2048, 3, seblock, reduction_ratio, dropout_rate, stride=2)
        
        # Remove global average pooling to preserve spatial dimension
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Conv1d(2048, output_dim, kernel_size=1)
        
    def _make_layer(self, in_channels, bottleneck_channels, out_channels, num_blocks, 
                    seblock, reduction_ratio, dropout_rate, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(BottleneckBlock1D(in_channels, bottleneck_channels, out_channels, 
                                      stride, downsample, seblock, reduction_ratio, dropout_rate))
        
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock1D(out_channels, bottleneck_channels, out_channels,
                                          seblock=seblock, reduction_ratio=reduction_ratio, 
                                          dropout_rate=dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: [num_curves, num_points, channels]
        # Conv1D expects: [batch_size, channels, sequence_length]
        x = x.transpose(1, 2)  # [num_curves, channels (latent_dim), num_points]
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
       
        x = self.dropout(x)
        x = self.fc(x)  # [num_curves, output_dim, num_points]
        
        # Transpose back to [num_curves, num_points, output_dim]
        x = x.transpose(1, 2)
        
        return x
        
class H_theta(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=5, num_neurons=512):
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
    
