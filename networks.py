import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, latent_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_size=32, hidden_size=128, output_size=2):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.decoder(x)


class ContractionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # possibly double number of channels here
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # halve size of image
        )
    
    def forward(self, x):
        return self.contract(x)

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # possibly halve number of channels here
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),  # double size of image
        )
    
    def forward(self, x):
        return self.expand(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, image_width=32, image_height=32):
        super().__init__()
        # TODO (student): If you want to use a UNet, you may use this class
        
        self.contractor = nn.Sequential(       # 32 x 32
            ContractionBlock(in_channels, 64), # 16 x 16
            ContractionBlock(64, 128),        # 8 x 8
            ContractionBlock(128, 256)         # 4 X 4
        )
        
        self.expander = nn.Sequential(       # 32 x 32
            ExpansionBlock(256, 128), # 16 x 16
            ExpansionBlock(128, 64),        # 8 x 8
            ExpansionBlock(64, out_channels),         # 4 X 4
        )
        
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        
        # TODO (student): If you want to use a UNet, you may use this class
        outputs = self.contractor(outputs)
        outputs = self.expander(outputs)
        
        
        outputs = outputs.reshape(batch, -1)
        return outputs
