import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from torch.optim import lr_scheduler
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels=2, conv_layers=2, hidden_dim=64, 
                 activation_func=nn.ReLU, dropout_rate=0.2,
                 use_batch_norm=True, initial_kernel_size=3):
        
        super(ConvBlock, self).__init__()
        
        self.conv_layers = conv_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.activation = activation_func
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Flag to indicate if this is the final block in a network
        # Can be set after initialization if needed
        self.is_final_block = False
        
        # Create a ModuleList to hold variable number of conv layers
        layers = []
        
        # First conv layer (in_channels → hidden_dim)
        layers.append(nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=initial_kernel_size))
        
        # Add batch norm after conv but before activation
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_dim))
            
        layers.append(self.activation())
        
        # Add dropout after activation (with lower rate for conv layers)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout2d(self.dropout_rate))
        
        # Add remaining conv layers (hidden_dim → hidden_dim)
        for i in range(conv_layers - 1):
            layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(self.hidden_dim))
                
            layers.append(self.activation())
            
            # Add dropout after each activation except possibly the last one
            # For the last layer, only add dropout if this block is not the final output
            if self.dropout_rate > 0 and (i < conv_layers - 2 or not self.is_final_block):
                layers.append(nn.Dropout2d(self.dropout_rate))
        
        # Create sequential module from the list of layers
        self.conv_block = nn.Sequential(*layers)
        
    
    def forward(self, x):
        return self.conv_block(x)
    
    def set_as_final_block(self, is_final=True):
        """
        Set whether this block is the final block in the network.
        Affects dropout behavior in the last layer.
        """
        self.is_final_block = is_final


class Decoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # From 128×9×9 → 64×18×18
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            # Intermediate refinement (maintain 64 channels)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
            # From 64×18×18 → 32×36×36
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        
            # Refine spatial dimension: 36×36 → 32×32 smoothly
            nn.Conv2d(32, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        
            # Intermediate refinement: 32×32 → 16×32×32
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        
            # Final smooth transition: 16×32×32 → 8×32×32
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # Gentle final mapping: 8×32×32 → 4×32×32
            nn.Conv2d(8, 4, kernel_size=1),
            nn.Sigmoid()  # Assuming output values between 0 and 1
        )

    def forward(self,x):
        return self.decoder(x)
        
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 2):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.reduced_channels = max(1, in_channels // reduction_ratio)  # modest reduction

        self.query_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)

        # Output projection back to original dimensions
        self.out_proj = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter initialized to zero
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Compute Q, K, V
        query = self.query_conv(x).view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)  # B x N x C'
        key = self.key_conv(x).view(batch_size, self.reduced_channels, -1)  # B x C' x N
        value = self.value_conv(x).view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)  # B x N x C'

        # Scaled dot-product attention
        scale = (self.reduced_channels) ** -0.5
        attention = torch.bmm(query, key) * scale  # B x N x N
        attention = F.softmax(attention, dim=-1)

        # Apply attention weights
        out = torch.bmm(attention, value)  # B x N x C'
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.reduced_channels, H, W)

        # Project back to original channel dimension
        out = self.out_proj(out)

        # Stable residual connection
        out = self.gamma * out + x

        return out


class DualInputTopologyVectorFields(nn.Module):

    def __init__(self, in_channels=4, hidden_dim=64):
        super(DualInputTopologyVectorFields, self).__init__()

        self.topology = nn.Sequential(
            ConvBlock(initial_kernel_size=5),
            nn.MaxPool2d(2),
            ConvBlock(64),
            # nn.MaxPool2d(2)
        )

        self.vector_field = nn.Sequential(
            ConvBlock(initial_kernel_size=5),
            nn.MaxPool2d(2),
            ConvBlock(64),
            # nn.MaxPool2d(2)
        )

        # After two MaxPool2d operations, the spatial dimensions will be reduced by 4x
        # After both branches, we'll have two feature maps of size hidden_dim*2
        concat_dim = hidden_dim * 2 # 2*hidden_dim from each branch

        # Self-attention on the concatenated features
        self.attention = SelfAttentionBlock(concat_dim)
        
         # Decoder path (instead of self.combine)
        self.decoder = Decoder(hidden_dim=hidden_dim)
    
    def forward(self, topology_input, vector_field_input):
        topology_features = self.topology(topology_input)
        vector_field_features = self.vector_field(vector_field_input)

        #concatenate
        combined_features = torch.cat([topology_features, vector_field_features], dim=1)

        #self attention features
        attention_features = self.attention(combined_features)

        #Final
        output = self.decoder(attention_features)

        return output
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DualInputTopologyVectorFields(in_channels=4, hidden_dim=64).to(device)

    summary(model, [(2, 32, 32), (2, 32, 32)], batch_size=32)
