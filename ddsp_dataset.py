import os
import torch
import torchaudio
import numpy as np
from ddsp import freeverb
import time

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Set the paths
x_file = "balloon_burst.wav"
x_dir = "data/freeverb_dataset/x"
p_dir = "data/freeverb_dataset/p"
y_dir = "data/freeverb_dataset/y"

n_samples = 5

# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

# Define the Freeverb parameter space
n_c = 8
c = (
    (1000, 500),  # delay mean, stdev
    (0.75, 0.1),  # feedback gain mean, stdev
    (0.2, 0.05)  # damping mean, stdev
)

n_a = 4
a = (
    (500, 200),  # delay mean, stdev
    (0.5, 0.2)  # feedback gain mean, stdev
)

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the audio file
file_path = os.path.join(x_dir, x_file)
x, sr = torchaudio.load(file_path)
# Normalize the x to the range [-1, 1]
x /= torch.max(torch.abs(x))

start_time = time.time()

for i in range(n_samples):
    # Apply the Freeverb effect with randomized parameters
    cM = torch.round(torch.normal(c[0][0], c[0][1], size=(n_c,))).to(torch.int32)
    ca = torch.normal(c[1][0], c[1][1], size=(n_c,), dtype=torch.float32)
    cd = torch.normal(c[2][0], c[2][1], size=(n_c,), dtype=torch.float32)
    aM = torch.round(torch.normal(a[0][0], a[0][1], size=(n_a,))).to(torch.int32)
    aa = torch.normal(a[1][0], a[1][1], size=(n_a,), dtype=torch.float32)


    # Clip gain parameters to prevent exploding feedback
    ca = torch.clamp(ca, 0, 1)
    cd = torch.clamp(cd, 0, 1)
    aa = torch.clamp(aa, 0, 1)

    # Process the audio file with the Freeverb effect
    y = freeverb(
        x=x,
        cM=cM,
        ca=ca,
        cd=cd,
        aM=aM,
        aa=aa
    )

    # Store the parameters in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    p_path = os.path.join(p_dir, p_filename)
    p = torch.cat([cM, ca, cd, aM, aa])
    np.savetxt(p_path, p.numpy(), fmt='%.10f')

    # Save the processed x to a new audio file
    y = y.unsqueeze(1).cpu()
    y_filename = os.path.splitext(x_file)[0] + f"_{i}.wav"
    y_path = os.path.join(y_dir, y_filename)
    torchaudio.save(y_path, y, sr)
    print(f"Generated: {i + 1}/{n_samples}")

print(f"{n_samples} samples generated.")
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")