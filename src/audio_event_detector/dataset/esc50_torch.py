from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_event_detector.features.spectrogram import load_audio, mel_spectrogram


class ESC50Torch(Dataset):

    def __init__(self, root):
        self.root = Path(root)

        self.meta = pd.read_csv(self.root / "meta" / "esc50.csv")
        self.audio_dir = self.root / "audio"

        self.classes = sorted(self.meta["category"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):

        row = self.meta.iloc[idx]

        audio_path = self.audio_dir / row["filename"]
        label = row["category"]

        audio = load_audio(audio_path)
        spec = mel_spectrogram(audio)

        spec = torch.tensor(spec).unsqueeze(0).float()

        label_idx = self.class_to_idx[label]

        return spec, label_idx
