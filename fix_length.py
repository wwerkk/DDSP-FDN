import numpy as np, soundfile as sf
# trim/pad audio to 2s
time_s = 2
file = 'data/test/balloon_burst.wav'
x, sr = sf.read(file)
time = time_s * sr
print(len(x))
if len(x) > time:
    x = x[:time]
    print("sample trimmed")
    print(len(x))
    sf.write(file, x, sr)
elif len(x) < time:
    x = np.pad(x, (0, time - len(x)), 'constant', constant_values=(0))
    print("sample padded")
    print(len(x))
    sf.write(file, x, sr)
else:
    print("perfect length, nothing to do here!")