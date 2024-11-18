import os
import numpy as np
import torch
import torch.nn as nn  
from   torch.utils.data import Dataset
# ------------------------------------------------------------------------
# AI Model Definition
# ------------------------------------------------------------------------
class OPL3Model(nn.Module):
    def __init__(self):
        super(OPL3Model, self).__init__()        
        # Convolutional layers for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU()
        )

        # Pooling for downsampling
        self.pooling = nn.MaxPool1d(kernel_size=4, stride=4)

        # Attention for global feature extraction
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Fully connected layers for synthesizer configuration output
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (2048 // 4), 1024),  # Adjust input size based on pooling
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 222)  # Final output for synthesizer configuration
        )

    def forward(self, x):
        # Reshape for Conv1D
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, 2048]

        # Convolutional layers
        x = self.conv_layers(x)  # Shape: [batch_size, 64, 2048]

        # Pooling
        x = self.pooling(x)  # Shape: [batch_size, 64, 2048 // 4]

        # Attention
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 2048 // 4, 64]
        x, _ = self.attention(x, x, x)  # Self-attention

        # Flatten and fully connected layers
        x = x.reshape(x.size(0), -1)  # Shape: [batch_size, 64 * (2048 // 4)]
        x = self.fc_layers(x)  # Shape: [batch_size, 222]

        return x
# ------------------------------------------------------------------------
# Dataset Class - gets individual records from the training files

class OPL3Dataset(Dataset):
    def __init__(self, spect_file, reg_file):
        # input spectra
        self.spect_file = spect_file
        # output synth configs
        self.reg_file = reg_file
        # size of one frequency spectrum record (uint8[2048])
        self.spect_size = 2048  
        # size of one synthesizer configuration vector (float32[222])
        self.reg_size = 222 * 4  # 4 bytes per float
        # Get the number of available training samples
        self.num_samples = os.path.getsize(spect_file) // self.spect_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read only the required spectrum data from file
        with open(self.spect_file, 'rb') as f:
            f.seek(idx * self.spect_size)
            spect = np.frombuffer(f.read(self.spect_size), dtype=np.uint8).astype(np.float32) / 255.0

        # Read only the required reg data from file
        with open(self.reg_file, 'rb') as f:
            f.seek(idx * self.reg_size)
            reg = np.frombuffer(f.read(self.reg_size), dtype=np.float32)

        spect = torch.tensor(spect, dtype=torch.float32)
        reg = torch.tensor(reg, dtype=torch.float32)
        return spect, reg

'''
# This dataset gets the WHOLE FILE into RAM!

class OPL3Dataset(Dataset):
    def __init__(self, spect_file, reg_file):
        self.spect_file = spect_file
        self.reg_file = reg_file
        self.spect_data = np.fromfile(spect_file, dtype=np.uint8).reshape(-1, 2048) / 255.0
        self.reg_data = np.fromfile(reg_file, dtype=np.float32).reshape(-1, 222)

    def __len__(self):
        return len(self.spect_data)

    def __getitem__(self, idx):
        spect = torch.tensor(self.spect_data[idx], dtype=torch.float32)
        reg = torch.tensor(self.reg_data[idx], dtype=torch.float32)
        return spect, reg
'''
