import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.optim import Adam
import torchaudio
import numpy as np
import os
import torch.nn.functional as F

impulse_file = 'data/freeverb_dataset/x/balloon_burst.wav'
dataset_dir = 'data/freeverb_dataset'
y_dir = os.path.join(dataset_dir, 'y')
params_dir = os.path.join(dataset_dir, 'p_n')
mfcc_dir = os.path.join(dataset_dir, 'mfcc_n')

audio_extension = '.wav'
param_extension = '.txt'
mfcc_extension = '.txt'

epochs = 100
batch_size = 16

trunc_len_s = 2
sr = 44100
trunc_len = 2 * sr

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'Device: {device}')

class AudioDataset(Dataset):
    def __init__(self, y_dir, params_dir, mfcc_dir, audio_extension='.wav', param_extension='.txt'):
        self.y_dir = y_dir
        self.params_dir = params_dir
        self.mfcc_dir = mfcc_dir
        self.audio_extension = audio_extension
        self.param_extension = param_extension
        self.mfcc_extension = mfcc_extension
        self.file_names = [f.replace(self.audio_extension, '') for f in os.listdir(y_dir) if f.endswith(self.audio_extension)]
        print(f"Loaded files: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # y_path = os.path.join(self.y_dir, file_name + self.audio_extension)
        params_path = os.path.join(self.params_dir, file_name + self.param_extension)
        mfcc_path = os.path.join(self.mfcc_dir, file_name + self.mfcc_extension)

        # Load the wet audio file
        # waveform, sample_rate = torchaudio.load(y_path)
        # waveform = waveform.squeeze(0)
        # waveform = waveform[:trunc_len]

        # Load MFCCs
        mfccs = np.loadtxt(mfcc_path)

        new_dimension = mfccs.shape[-1] // 16
        mfccs = mfccs.reshape(mfccs.shape[:-1] + (new_dimension, 16))
        # print("MFCC SHAPE")
        # print(mfccs.shape)
        mfccs = np.transpose(mfccs, axes=(1, 0))
        # print(mfccs.shape)

        # Load the parameters
        params = np.loadtxt(params_path)

        # Normalize the parameters
        params[:8] = self.normalize_freq(params[:8])
        params[16:20] = self.normalize_freq(params[24:28])

        return mfccs, params

    def normalize_freq(self, params):
        # Convert params to a numpy array
        params = np.array(params)

        # Perform normalization for 20hz-20khz
        normalized_params = (params - 20) / (20000 - 20)

        # Return the normalized params
        return normalized_params

    def denormalize_freq(self, normalized_params):
        # Perform denormalization using the saved scaling factors
        denormalized_params = normalized_params * (20000 - 20) + 20

        # Return the denormalized params
        return denormalized_params

class AudioModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AudioModel, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dense3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.dense1(x))
        out, _ = self.gru1(out)
        out = F.relu(self.dense2(out))
        out, _ = self.gru2(out)
        out = self.dense3(out)
        return out

# Load the dataset
dataset = AudioDataset(y_dir, params_dir, mfcc_dir, audio_extension, param_extension)

# Specify the validation split ratio (e.g., 0.2 for 20% validation data)
validation_split = 0.2

# Compute the split indices
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Create the training and validation samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create the training and validation data loaders
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# impulse, sr = torchaudio.load(impulse_file)
# resampler = T.Resample(sr, 16000, dtype=impulse.dtype)
# waveform = resampler(impulse)
# sr = 16000

# Get a sample input from the dataset
sample_data = next(iter(train_dataloader))
sample_input = sample_data[0]  # Assuming the input is the first element in the sample data

# Calculate the input size
input_size = tuple(sample_input.shape[1:])
input_size = input_size[0] * input_size[1]
input_size = 345

print("Input size:", input_size)

# Define the model
model = AudioModel(input_size, 16, 32)  # Assuming parameters are 32-dimensional
optimizer = Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

best_val_loss = float('inf')
validate_every = 10  # Perform validation every 5 epochs

#train model
for epoch in range(epochs):
    train_loss = 0.0

    # Training
    model.train()
    for batch in train_dataloader:
        mfccs, params = batch
        mfccs = mfccs.float()
        params = params.float()  # Convert to Float
        # Forward pass
        outputs = model(mfccs)
        # Compute the loss between the model's output and the parameters
        train_loss = criterion(outputs, params)

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # Compute average training loss
    train_loss /= len(train_dataloader)

    # Validation
    if (epoch + 1) % validate_every == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                mfccs, params = batch
                mfccs = mfccs.float()
                params = params.float()  # Convert to Float
                # Forward pass
                outputs = model(mfccs)
                # TODO fix the shape difference issues between outputs and targets
                # Compute the loss between the model's output and the parameters
                print(outputs.shape)
                print(outputs)
                print(params.shape)
                print(params)
                loss = criterion(outputs, params)

                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(val_dataloader)

        # Update best validation loss and save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}   Val Loss: {val_loss}')
    else:
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}')