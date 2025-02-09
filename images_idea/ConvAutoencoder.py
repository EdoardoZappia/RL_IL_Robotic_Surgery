import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Encoder, self).__init__()
        
        # Primo livello di convoluzione
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Secondo livello di convoluzione
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Terzo livello di convoluzione
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, output_channels=3):
        super(Decoder, self).__init__()

        # Secondo livello di deconvoluzione
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Terzo livello di deconvoluzione
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Quarto livello di deconvoluzione
        self.up3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))

        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(output_channels=input_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Esempio di utilizzo
if __name__ == "__main__":
    autoencoder = Autoencoder()
    print(autoencoder)

