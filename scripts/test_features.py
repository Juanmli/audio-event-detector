from audio_event_detector.dataset.esc50 import ESC50Dataset
from audio_event_detector.features.spectrogram import load_audio, mel_spectrogram

dataset = ESC50Dataset("data/raw/esc50")

path = dataset.get_audio_path(0)

audio = load_audio(path)

spec = mel_spectrogram(audio)

print("audio length:", len(audio))
print("spectrogram shape:", spec.shape)

import matplotlib.pyplot as plt

plt.imshow(spec, aspect="auto", origin="lower")
plt.colorbar()
plt.show()
