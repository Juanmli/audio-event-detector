import sys
import torch

from audio_event_detector.models.cnn import AudioCNN
from audio_event_detector.features.spectrogram import load_audio, mel_spectrogram

classes = ['cat', 'crying_baby', 'dog', 'glass_breaking', 'siren']

model = AudioCNN(num_classes=len(classes))
model.load_state_dict(torch.load("model_best.pt"))
model.eval()

wav_path = sys.argv[1]

audio = load_audio(wav_path)
spec = mel_spectrogram(audio)

spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0).float()

with torch.no_grad():
    output = model(spec)
    probs = torch.softmax(output, dim=1)

pred = torch.argmax(probs, dim=1).item()

print("prediction:", classes[pred], probs[0, pred].item())
