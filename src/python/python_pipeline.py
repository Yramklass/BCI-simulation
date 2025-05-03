import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model("best_attention_model.keras", compile=False)

# Load incoming CSV
df = pd.read_csv("test_input.csv")  # Replace with your actual input file

# Filter for 'left' and 'right' labels only (if labels exist)
if 'label' in df.columns:
    df = df[df['label'].isin(['left', 'right'])]
    df['label'] = df['label'].map({'left': 0, 'right': 1})

# Define EEG channels in same order as training
channel_order = ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz'] + \
                [col for col in df.columns if col.startswith("EEG") and col not in ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz']]
eeg_channels = [col for col in channel_order if col in df.columns]

# Patient-wise z-score normalization (if 'patient' field exists)
if 'patient' in df.columns:
    for patient in df['patient'].unique():
        mask = df['patient'] == patient
        for channel in eeg_channels:
            df.loc[mask, channel] = (df.loc[mask, channel] - df.loc[mask, channel].mean()) / (df.loc[mask, channel].std() + 1e-8)
else:
    for channel in eeg_channels:
        df[channel] = (df[channel] - df[channel].mean()) / (df[channel].std() + 1e-8)

# Extract signal and delta features
signals = df[eeg_channels].values
deltas = np.diff(signals, axis=0, prepend=signals[0:1])
features = np.concatenate([signals, deltas], axis=1)  # (T, 44)

# Windowing: generate overlapping windows of shape (128, 44)
window_size = 128
step_size = 32
windows = []

for start in range(0, len(features) - window_size + 1, step_size):
    windows.append(features[start:start+window_size])

X = np.array(windows)

# Predict
probs = model.predict(X)
preds = (probs > 0.5).astype(int)

# Show output
print(f"Model predictions (0 = left, 1 = right):\n{preds.flatten()}")
print(f"Probabilities:\n{probs.flatten()}")
