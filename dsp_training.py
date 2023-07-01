import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.optim import Adam
import torchaudio
import numpy as np
import os
import torchaudio.transforms as T, torch.nn.functional as F

# Create MFCC transform
mfcc_transform = T.MFCC(
    sample_rate=44100,
    n_mfcc=13,
    melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 40, 'center': False}
)

impulse_file = 'data/freeverb_dataset/dry/balloon_burst_1.wav'
dataset_dir = 'data/freeverb_dataset'
wet_audio_dir = os.path.join(dataset_dir, 'wet')
params_dir = os.path.join(dataset_dir, 'params')

audio_extension = '.wav'
param_extension = '.txt'

epochs = 100
batch_size = 8

class AudioDataset(Dataset):
    def __init__(self, wet_audio_dir, params_dir, audio_extension='.wav', param_extension='.txt'):
        self.wet_audio_dir = wet_audio_dir
        self.params_dir = params_dir
        self.audio_extension = audio_extension
        self.param_extension = param_extension
        self.file_names = [f.replace(self.audio_extension, '') for f in os.listdir(wet_audio_dir) if f.endswith(self.audio_extension)]
        print(f"Loaded files: {len(self.file_names)}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        wet_audio_path = os.path.join(self.wet_audio_dir, file_name + self.audio_extension)
        params_path = os.path.join(self.params_dir, file_name + self.param_extension)

        # Load the wet audio file
        waveform, sample_rate = torchaudio.load(wet_audio_path)
        waveform = waveform.squeeze(0)

        # Compute MFCCs from the wet audio
        mfccs = mfcc_transform(waveform)

         # Normalize the MFCCs
        mfccs = F.normalize(mfccs)

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

import torch.nn.functional as F

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
dataset = AudioDataset(wet_audio_dir, params_dir, audio_extension, param_extension)
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

impulse, sr = torchaudio.load(impulse_file)
# resampler = T.Resample(sr, 16000, dtype=impulse.dtype)
# waveform = resampler(impulse)
# sr = 16000

# Define the model
model = AudioModel(1079, 64, 28)  # Assuming audio files are 1 second at 44100Hz and parameters are 24-dimensional
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
        wet_audio, params = batch
        wet_audio = wet_audio.view(wet_audio.size(0), -1).float()  # Convert to Float
        params = params.float()  # Convert to Float

        # Forward pass
        outputs = model(wet_audio)

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
                wet_audio, params = batch
                wet_audio = wet_audio.view(wet_audio.size(0), -1).float()  # Convert to Float
                params = params.float()  # Convert to Float

                # Forward pass
                outputs = model(wet_audio)

                # Compute the loss between the model's output and the parameters
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




# for epoch in range(epochs):  # 100 epochs
#     for batch in dataloader:
#         wet_audio, params = batch
#         wet_audio = wet_audio.view(wet_audio.size(0), -1)  # Flatten the audio data

#         # Forward pass
#         outputs = model(wet_audio)

#         # Use the model's output to generate parameters for the freeverb function
#         processed_waveforms = torch.empty(32, sr)
#         for i, sample in enumerate(outputs):
#             params = sample.detach()
#             # print(params.shape)
#             # Process the impulse audio file with the Freeverb effect
#             impulse_waveform = impulse.squeeze(0)
#             print(f"Applying reverb to sample {i} in batch")
#             processed_waveform = freeverb(impulse_waveform, params)
#             processed_waveforms[i] = processed_waveform

#         print(f"Reverb applied to entire batch, calculating loss...")
#         # Compute the loss between the processed waveform and the target waveform
#         loss = criterion(processed_waveforms, wet_audio)

#         # Backward pass and optimization
#         print(f"Backward pass...")
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

    # print(f'Epoch {epoch+1}, Loss: {loss.item()}')