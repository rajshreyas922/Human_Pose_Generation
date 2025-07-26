import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock1D_LayerNorm(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, 
                 stride=1, downsample=None, seblock=True, reduction_ratio=0.25,
                 dropout_rate=0.1):
        super().__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.ln1 = LayerNorm1d(bottleneck_channels)
        
        # 1x1 conv (main processing)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=1, 
                              stride=stride, bias=False)
        self.ln2 = LayerNorm1d(bottleneck_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.ln3 = LayerNorm1d(out_channels)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)  # Keep LeakyReLU for now
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # SE block
        self.seblock = SEBlock1D_Conv_Global(out_channels, reduction_ratio) if seblock else nn.Identity()
        
        # Keep learnable residual scale for now
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.leaky_relu(out)
        
        out = self.conv3(out)
        out = self.ln3(out)
        
        # Apply SE block
        out = self.seblock(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Use learnable residual scale
        out = out * self.res_scale + identity
        out = self.leaky_relu(out)
        
        return out

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x):
        # x: [batch, channels, length]
        mean = x.mean(dim=1, keepdim=True)  # [batch, 1, length]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [batch, 1, length]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters
        gamma = self.gamma.view(1, -1, 1)  # [1, channels, 1]
        beta = self.beta.view(1, -1, 1)    # [1, channels, 1]
        
        return x_norm * gamma + beta


class H_theta_ResNet1D_Like_Old(nn.Module):
    def __init__(self, input_dim, output_dim=3, 
                 hidden_dims=[256, 512, 1024, 2048],
                 bottleneck_dims=[64, 128, 256, 512],  # New: bottleneck dimensions
                 blocks_per_dim=[3, 4, 6, 3], 
                 dropout_rate=0.2, seblock=True, reduction_ratio=0.15,
                 use_bottleneck=True):  # Option to toggle bottleneck
        super().__init__()
        self.seblock_enabled = seblock
        self.use_bottleneck = use_bottleneck
        
        if isinstance(blocks_per_dim, int):
            blocks_per_dim = [blocks_per_dim] * len(hidden_dims)
        
        # Ensure bottleneck dims match hidden dims
        if use_bottleneck and len(bottleneck_dims) != len(hidden_dims):
            # Auto-generate bottleneck dims as 1/4 of hidden dims
            bottleneck_dims = [h // 4 for h in hidden_dims]
        
        # Input layer
        # self.input_layer = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_dims[0], kernel_size=1),
        #     LayerNorm1d(hidden_dims[0]),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(dropout_rate)
        # )
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims[0]//2, kernel_size=1),
            LayerNorm1d(hidden_dims[0]//2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dims[0]//2, hidden_dims[0], kernel_size=1),
            LayerNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for stage_idx, (next_dim, num_blocks) in enumerate(zip(hidden_dims, blocks_per_dim)):
            blocks = nn.ModuleList()
            
            for block_idx in range(num_blocks):
                # First block in stage handles dimension change
                if block_idx == 0 and current_dim != next_dim:
                    downsample = nn.Sequential(
                        nn.Conv1d(current_dim, next_dim, kernel_size=1, bias=False),
                        LayerNorm1d(next_dim)
                    )
                else:
                    downsample = None
                
                if use_bottleneck:
                    # Use bottleneck block
                    block = BottleneckBlock1D_LayerNorm(
                        in_channels=current_dim if block_idx == 0 else next_dim,
                        bottleneck_channels=bottleneck_dims[stage_idx],
                        out_channels=next_dim,
                        downsample=downsample,
                        seblock=seblock,
                        reduction_ratio=reduction_ratio,
                        dropout_rate=dropout_rate
                    )
                else:
                    # Use simple block (your current implementation)
                    if downsample is not None:
                        # Need to handle dimension change
                        block = nn.ModuleDict({
                            'block': nn.Sequential(
                                nn.Conv1d(current_dim, next_dim, kernel_size=1),
                                LayerNorm1d(next_dim),
                                nn.LeakyReLU(0.2),
                                nn.Dropout(dropout_rate),
                                nn.Conv1d(next_dim, next_dim, kernel_size=1),
                                LayerNorm1d(next_dim),
                            ),
                            'downsample': downsample,
                            'se_block': SEBlock1D_Conv_Global(next_dim, reduction_ratio) if seblock else None,
                            'res_scale': nn.Parameter(torch.tensor(0.5))
                        })
                    else:
                        block = nn.ModuleDict({
                            'block': nn.Sequential(
                                nn.Conv1d(next_dim, next_dim, kernel_size=1),
                                LayerNorm1d(next_dim),
                                nn.LeakyReLU(0.2),
                                nn.Dropout(dropout_rate),
                                nn.Conv1d(next_dim, next_dim, kernel_size=1),
                                LayerNorm1d(next_dim),
                            ),
                            'downsample': None,
                            'se_block': SEBlock1D_Conv_Global(next_dim, reduction_ratio) if seblock else None,
                            'res_scale': nn.Parameter(torch.tensor(0.5))
                        })
                
                blocks.append(block)
                
                if block_idx == 0 and current_dim != next_dim:
                    current_dim = next_dim
            
            self.stages.append(blocks)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer = nn.Conv1d(current_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        
        for stage in self.stages:
            for block in stage:
                if self.use_bottleneck:
                    x = block(x)
                else:
                    # Handle simple blocks
                    residual = x
                    if block['downsample'] is not None:
                        residual = block['downsample'](residual)
                    
                    out = block['block'](x)
                    if block['se_block'] is not None:
                        out = block['se_block'](out)
                    out = out * block['res_scale'] + residual
                    x = F.leaky_relu(out, 0.2)
        
        x = self.dropout(x)
        x = self.final_layer(x)
        x = x.transpose(1, 2)
        
        return x

class H_theta_Res_old(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=1024, num_blocks = 8, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(num_blocks) 
        ])

        self.res_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(len(self.res_blocks))
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
                 dropout_rate=0.2, seblock=True, reduction_ratio=0.15):
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

class SEBlock1D_Conv_Global(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        self.squeeze_excite = nn.Sequential(
            nn.Conv1d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [num_curves, channels, num_points]
        # Global pooling across points (not channels!)
        squeezed = x.mean(dim=2, keepdim=True)  # [num_curves, channels, 1]
        se_weights = self.squeeze_excite(squeezed)
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
        #self.seblock = SEBlock1D_Conv(out_channels, reduction_ratio) if seblock else nn.Identity()
        self.seblock = SEBlock1D_Conv_Global(out_channels, reduction_ratio) if seblock else nn.Identity()

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
                 seblock=True, reduction_ratio=0.1):
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
        
        
        # Sticks
        # self.layer1 = self._make_layer(64, 96, 1024, 2, seblock, reduction_ratio, dropout_rate)
        # self.layer2 = self._make_layer(384, 192, 1024, 3, seblock, reduction_ratio, dropout_rate, stride=2)
        # self.layer3 = self._make_layer(768, 384, 1536, 4, seblock, reduction_ratio, dropout_rate, stride=2)
        # self.layer4 = self._make_layer(1536, 768, 3072, 2, seblock, reduction_ratio, dropout_rate, stride=2)

        # Sticks
        # self.layer1 = self._make_layer(64, 64, 1024, 3, seblock, reduction_ratio, dropout_rate)
        # self.layer2 = self._make_layer(1024, 128, 1024, 3, seblock, reduction_ratio, dropout_rate, stride=2)
        # self.layer3 = self._make_layer(1024, 256, 1024, 6, seblock, reduction_ratio, dropout_rate, stride=2)  # Using 6 instead of 23 for computational efficiency
        # self.layer4 = self._make_layer(1024, 512, 2048, 3, seblock, reduction_ratio, dropout_rate, stride=2)

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
    
