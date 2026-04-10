import librosa
import numpy as np
from tensorflow.keras.utils import Sequence


class AudioDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32,
                 sr=22050, duration=3, n_mels=128,
                 shuffle=True, augment=False):

        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.shuffle = shuffle
        self.augment = augment

        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []

        for i in batch_indices:
            file_path = self.file_paths[i]
            label = self.labels[i]

            try:
                audio, _ = librosa.load(file_path, sr=self.sr)

                if self.augment:
                    audio = self.augment_audio(audio)

                max_len = self.sr * self.duration
                if len(audio) < max_len:
                    audio = np.pad(audio, (0, max_len - len(audio)))
                else:
                    audio = audio[:max_len]

                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.sr,
                    n_mels=self.n_mels
                )

                mel = librosa.power_to_db(mel, ref=np.max)

                mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)

                mel = mel[..., np.newaxis]

                X.append(mel)
                y.append(label)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def augment_audio(self, audio):
        if np.random.rand() < 0.3:
            audio += 0.005 * np.random.randn(len(audio))

        if np.random.rand() < 0.3:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=2)

        if np.random.rand() < 0.3:
            audio = librosa.effects.time_stretch(audio, rate=1.1)

        return audio