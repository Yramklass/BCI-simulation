import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
df = pd.read_csv("BCICIV_2a_all_patients.csv")  

# Filter labels: only keep 'left' and 'right'
df = df[df['label'].isin(['left', 'right'])]
df['label'] = df['label'].map({'left': 0, 'right': 1})
print("Original class balance (after filtering):\n", df['label'].value_counts())

# Extract EEG channels (everything except metadata columns)
eeg_channels = [col for col in df.columns if col.startswith("EEG")]
print(f"Present EEG channels: {eeg_channels}")

# Parameters
window_size = 64
step_size = 32  # Overlap
X_windows, y_windows = [], []

# Group by epoch and window each one
for epoch_id, epoch_df in df.groupby("epoch"):
    signals = epoch_df[eeg_channels].values
    labels = epoch_df["label"].values
    for start in range(0, len(signals) - window_size + 1, step_size):
        end = start + window_size
        window = signals[start:end]
        label = np.bincount(labels[start:end]).argmax()  # majority label
        X_windows.append(window)
        y_windows.append(label)

X = np.array(X_windows)
y = np.array(y_windows)
print(f"\nNumber of windows created: {len(X)}")
print(f"Shape of each window: {X[0].shape}")
print(f"Class balance after windowing:\n{dict(pd.Series(y).value_counts())}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Normalize
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# CNN-LSTM model with Cosine LR decay
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model(X_train.shape[1:])

# Cosine decay LR
initial_lr = 2e-4
cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=10 * len(X_train) // 32,
)

optimizer = optimizers.Adam(learning_rate=cosine_lr)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks
ckpt_path = "best_model.keras"
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# Training
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=callbacks_list
)

# Evaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nEvaluation:")
print(classification_report(y_test, y_pred, digits=4))

# Save model
model.save("temp_model.keras")  # Recommended format in Keras 3
print("Model saved as temp_model.keras")

# ONNX conversion
print("⚠️ To convert to ONNX, run:\n")
print("   python -m tf2onnx.convert --keras temp_model.keras --output eeg_cnn_model.onnx")
