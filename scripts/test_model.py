import torch

from audio_event_detector.models.cnn import AudioCNN

model = AudioCNN(num_classes=50)

x = torch.randn(8, 1, 64, 250)

y = model(x)

print("output shape:", y.shape)
