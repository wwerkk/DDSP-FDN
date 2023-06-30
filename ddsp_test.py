from ddsp import freeverb
import torch
from torchaudio import load, save

# Load the audio file
file_path = 'data/freeverb_dataset/dry/balloon_burst_1.wav'
waveform, fs = load(file_path)

# Convert the waveform to a PyTorch tensor
waveform = waveform.squeeze(0)  # Remove the channel dimension if present
waveform = waveform.to(torch.float32)  # Convert to float32 if necessary

# Normalize the waveform to the range [-1, 1]
waveform /= torch.max(torch.abs(waveform))

# Apply the differentiable Freeverb to the waveform
processed_waveform = freeverb(waveform, fs)

# Reshape the processed waveform to have two dimensions
processed_waveform = processed_waveform.unsqueeze(0)  # Add the channel dimension back

# Convert the processed waveform back to a NumPy array
processed_waveform = processed_waveform.cpu()

# Save the processed waveform as an audio file
output_path = 'data/freeverb_dataset/wet/balloon_burst_1.wav'
save(output_path, processed_waveform, fs)
