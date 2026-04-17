from audio_event_detector.dataset.esc50 import ESC50Dataset

dataset = ESC50Dataset("data/raw/esc50")

print("dataset size:", len(dataset))

print(dataset.get_audio_path(0))
print(dataset.get_label(0))
