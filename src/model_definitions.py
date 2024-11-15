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
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),  # First conv layer
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),  # Second conv layer
            nn.ReLU()
        )

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Fully connected layers
        self.model = nn.Sequential(
            nn.Linear(64 * 2048, 2048),  # Adjusted input size for flattened Conv1d output
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 222)
        )
    def forward(self, x):
        # Reshape input to add a channel dimension for Conv1d layers
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, 2048]  
        # Pass through convolutional layers
        x = self.conv_layers(x) 
        
        # Prepare for attention by permuting to (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # Shape becomes (batch_size, 2048, 32)

        # Apply multi-head attention (self-attention) where query, key, and value are all `x`
        attn_output, _ = self.attention(x, x, x)  # Shape: (batch_size, 2048, 32)
        
        # Flatten the output from the attention layer for dense layers
        attn_output = attn_output.reshape(attn_output.size(0), -1)  # Shape: (batch_size, 32 * 2048)

        # Flatten the output from the Conv1d layers
        #x = x.view(x.size(0), -1)  # Shape becomes [batch_size, 32 * 2048]        
        # Pass through the fully connected layers
        x = self.model(attn_output)        
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
