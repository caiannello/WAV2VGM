# ------------------------------------------------------------------------
'''
Release notes per ChatGPT30, who ported this code 
from the TF version for me:

Key Adjustments

Data Loading: I used a CustomDataset to handle binary data
loading and parsing, similar to your TensorFlow code. 

Model Definition: The architecture is recreated using 
torch.nn.Sequential with nn.BatchNorm1d for batch normalization 
and nn.ReLU for activation functions.

Training and Validation Loop: Includes manual training and 
validation steps with early stopping.

Early Stopping: Saves the best model based on validation loss.

Sanity Check: Grabs a batch from val_loader to compare 
predictions with true outputs.

Plotting: Same loss plotting as in your TensorFlow code.

This code should give you comparable results in PyTorch and allow 
you to leverage GPU acceleration effectively. Let me know if you 
need further assistance with this transition!

'''
# ------------------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------

# Device configuration
devname = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CUDA or CPU? I'm selecting: ---> {devname.upper()} <---")

device = torch.device(devname)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Parameters
batch_size = 32
fname0 = dir_path+'/../training_sets/opl3_training_spects.bin'
fname1 = dir_path+'/../training_sets/opl3_training_regs.bin'
epochs = 50
early_stopping_patience = 5

# ------------------------------------------------------------------------
# Custom Dataset Class
'''
# reads whole files into ram
class CustomDataset(Dataset):
    def __init__(self, spect_file, reg_file):
        self.spect_file = spect_file
        self.reg_file = reg_file
        self.spect_data = np.fromfile(spect_file, dtype=np.uint8).reshape(-1, 2048) / 255.0
        self.reg_data = np.fromfile(reg_file, dtype=np.float32).reshape(-1, 290)

    def __len__(self):
        return len(self.spect_data)

    def __getitem__(self, idx):
        spect = torch.tensor(self.spect_data[idx], dtype=torch.float32)
        reg = torch.tensor(self.reg_data[idx], dtype=torch.float32)
        return spect, reg
'''
# getter gets individual records from files
class CustomDataset(Dataset):
    def __init__(self, spect_file, reg_file):
        self.spect_file = spect_file
        self.reg_file = reg_file
        # Get the number of samples by calculating the file size
        self.spect_size = 2048  # size of one spectrum entry in bytes
        self.reg_size = 290 * 4  # size of one reg entry in bytes (float32 is 4 bytes)
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

print('\nTODO: Verify the spect/reg training sets are in-sync throughout.\n')

# Load dataset and split into training and validation sets
full_dataset = CustomDataset(fname0, fname1)
dataset_size = len(full_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------
'''
# this version of the model lacks convolutional leyers, and it doesnt 
# seem too reach an acceptable result after a lot of training.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 290)
        )

    def forward(self, x):
        return self.model(x)
'''

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # First conv layer
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # Second conv layer
            nn.ReLU()
        )        
        # Fully connected layers
        self.model = nn.Sequential(
            nn.Linear(32 * 2048, 2048),  # Adjusted input size for flattened Conv1d output
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 290)
        )
    def forward(self, x):
        # Reshape input to add a channel dimension for Conv1d layers
        x = x.unsqueeze(1)  # Shape becomes [batch_size, 1, 2048]        
        # Pass through convolutional layers
        x = self.conv_layers(x)        
        # Flatten the output from the Conv1d layers
        x = x.view(x.size(0), -1)  # Shape becomes [batch_size, 32 * 2048]        
        # Pass through the fully connected layers
        x = self.model(x)        
        return x

model = Model().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------------------
# Training and Validation
train_losses, val_losses = [], []
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print status update every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                  f"Batch Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            running_val_loss += val_loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), dir_path+'/../models/torch_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load(dir_path+'/../models/torch_model.pth'))

# ------------------------------------------------------------------------
# Sanity Check
print("Sanity check:")
val_loader_iter = iter(val_loader)
for sample_input, true_output in val_loader_iter:
    sample_input, true_output = sample_input.to(device), true_output.to(device)
    predicted_output = model(sample_input)
    print("Sample Input:", sample_input.cpu().numpy())
    print("Predicted Output:", predicted_output.cpu().numpy())
    print("True Output:", true_output.cpu().numpy())
    break

# ------------------------------------------------------------------------
# Plotting Loss Graph
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

print('Program END.')

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
