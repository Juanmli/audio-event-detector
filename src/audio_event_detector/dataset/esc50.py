from pathlib import Path
import pandas as pd

class ESC50Dataset:
    
    def __init__(self, root):
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "meta" / "esc50.csv")
        self.audio_dir = self.root / "audio"

    def __len__(self):
        return len(self.meta)

    def get_audio_path(self, idx):
        row = self.meta.iloc[idx]
        return self.audio_dir / row["filename"]

    def get_label(self, idx):
        row = self.meta.iloc[idx]
        return row["category"]
