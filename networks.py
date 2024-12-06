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
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # halve size of image
    
    def forward(self, x):
        skip = self.contract(x)
        output = self.pool(skip)
        return output, skip

class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # possibly halve number of channels here
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, int(out_channels/2), kernel_size=2, stride=2),  # double size of image
        )
    
    def forward(self, x):
        return self.expand(x)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # possibly double number of channels here
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)  # up-conv, halve e number of channels and double image size
        )
    
    def forward(self, x):
        return self.bottle(x)

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # possibly halve number of channels here
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1), # Convolve with 1x1 kernel
        )
    
    def forward(self, x):
        return self.output(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, image_width=32, image_height=32):
        super().__init__()
        # TODO (student): If you want to use a UNet, you may use this class
        
        #(batch_size, in_channels, 32, 32)
        self.cont1 = ContractionBlock(in_channels, 64)
        self.cont2 = ContractionBlock(64, 128)   
        self.cont3 = ContractionBlock(128, 256)        
        
        self.bottleneck = BottleNeck(256, 512)  
        #(batch_size, 256 * 2, 8, 8)
                
        self.exp1 = ExpansionBlock(512, 256)
        self.exp2 = ExpansionBlock(256, 128)   
        self.output = OutputBlock(128, 64)
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        
        # TODO (student): If you want to use a UNet, you may use this class
        
        
        # Contracting path (down-sampling)
        x1, skipinp = self.cont1(inputs)  # 32x32 -> 16x16
        x2, skip1 = self.cont2(x1)      # 16x16 -> 8x8
        x3, skip2 = self.cont3(x2)      # 8x8 -> 4x4
        
        # Bottleneck
        bn = self.bottleneck(x3)  # 4x4 -> 2x2
        
        # Expanding path
        x4 = torch.cat((skip2, bn), dim=1) 
        x4 = self.exp1(x4)  
        
        x5 = torch.cat((skip1, x4), dim=1) 
        x5 = self.exp2(x5)
        
        out = torch.cat((skipinp, x5), dim=1)  
        out = self.output(out)
        
        # Output block
        outputs = out
        
        
        outputs = outputs.reshape(batch, -1)
        return outputs
    