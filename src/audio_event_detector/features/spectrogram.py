import librosa
import numpy as np


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    return audio


def mel_spectrogram(audio, sr=16000, n_mels=64):

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=n_mels
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    return S_db
