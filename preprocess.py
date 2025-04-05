import os
import librosa
import librosa.display
import numpy as np

FIXED_WIDTH = 400  # Adjust as needed

def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Convert to dB
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Resize spectrogram to a fixed width (pad or truncate)
    if spec_db.shape[1] < FIXED_WIDTH:
        # Pad if too short
        pad_width = FIXED_WIDTH - spec_db.shape[1]
        spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if too long
        spec_db = spec_db[:, :FIXED_WIDTH]

    return spec_db
def load_dataset(dataset_path, max_files=1000):
    spectrograms, labels = [], []
    files = [f for f in os.listdir(dataset_path) if f.endswith(".flac")][:max_files]
    
    for file in files:
        spec = audio_to_spectrogram(os.path.join(dataset_path, file))
        spectrograms.append(spec)
        labels.append(0 if "real" in file else 1)
    
    X = np.array(spectrograms)[..., np.newaxis]
    y = np.array(labels)
    return X, y

if __name__ == "__main__":
    dataset_folder = "<path-to-dataset-folder>"
    X, y = load_dataset(dataset_folder)
    np.save("X.npy", X)
    np.save("y.npy", y)
