import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32, l1=1e-5, l2=1e-4):
        """
        Initialize the autoencoder with input and encoding dimensions and regularization parameters.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        """
        Forward pass for the autoencoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def regularization_loss(self):
        """
        Calculate the elastic net regularization loss.
        """
        l1_loss = 0
        l2_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
        return self.l1 * l1_loss + self.l2 * l2_loss
    

    