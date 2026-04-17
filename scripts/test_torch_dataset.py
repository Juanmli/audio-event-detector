from torch.utils.data import DataLoader
from audio_event_detector.dataset.esc50_torch import ESC50Torch

dataset = ESC50Torch("data/raw/esc50")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

batch = next(iter(loader))

x, y = batch

print("batch x shape:", x.shape)
print("batch y shape:", y.shape)
