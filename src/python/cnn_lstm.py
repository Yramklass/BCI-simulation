import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Load and preprocess data
df = pd.read_csv("BCICIV_2a_all_patients.csv")

# Filter labels: only keep 'left' and 'right'
df = df[df['label'].isin(['left', 'right'])]
df['label'] = df['label'].map({'left': 0, 'right': 1})

# Reorder channels by importance (motor cortex first)
channel_order = ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz'] + \
                [col for col in df.columns if col.startswith("EEG") and col not in ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz']]
eeg_channels = [col for col in channel_order if col in df.columns]

# Patient-specific normalization
for patient in df['patient'].unique():
    patient_mask = df['patient'] == patient
    for channel in eeg_channels:
        df.loc[patient_mask, channel] = (df.loc[patient_mask, channel] - 
                                        df.loc[patient_mask, channel].mean()) / \
                                       (df.loc[patient_mask, channel].std() + 1e-8)

# Parameters
window_size = 128  # Increased window size
step_size = 32     # 75% overlap
X_windows, y_windows = [], []

# Add delta features and create windows
def add_delta_features(signals):
    deltas = np.diff(signals, axis=0)
    deltas = np.vstack([deltas[0:1], deltas])  # Pad first time step
    return np.concatenate([signals, deltas], axis=1)

for epoch_id, epoch_df in df.groupby("epoch"):
    signals = epoch_df[eeg_channels].values
    signals = add_delta_features(signals)  
    
    labels = epoch_df["label"].values
    for start in range(0, len(signals) - window_size + 1, step_size):
        end = start + window_size
        window = signals[start:end]
        label = np.bincount(labels[start:end]).argmax()  # majority label
        X_windows.append(window)
        y_windows.append(label)

X = np.array(X_windows)
y = np.array(y_windows)

# Train/test split with patient awareness
patient_ids = df['patient'].unique()
test_patient = patient_ids[-1]  # Hold out last patient for testing

test_mask = df['patient'] == test_patient
train_df = df[~test_mask]
test_df = df[test_mask]

# Recreate windows for train/test split
def create_windows(dataframe):
    windows, labels = [], []
    for epoch_id, epoch_df in dataframe.groupby("epoch"):
        signals = epoch_df[eeg_channels].values
        signals = add_delta_features(signals)
        epoch_labels = epoch_df["label"].values
        for start in range(0, len(signals) - window_size + 1, step_size):
            end = start + window_size
            window = signals[start:end]
            label = np.bincount(epoch_labels[start:end]).argmax()
            windows.append(window)
            labels.append(label)
    return np.array(windows), np.array(labels)

X_train, y_train = create_windows(train_df)
X_test, y_test = create_windows(test_df)

# Simplified model architecture
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
        
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model(X_train.shape[1:])

# Simplified learning rate schedule
initial_learning_rate = 1e-4
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=200 * len(X_train) // 32
)

# Callbacks
ckpt_path = "best_model.keras"
callbacks_list = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
]

# Class weights
class_weights = {0: len(y_train)/(2*np.sum(y_train==0)), 
                 1: len(y_train)/(2*np.sum(y_train==1))}

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name='auc')]
)

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=200,
    class_weight=class_weights,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nEvaluation Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save model
model.save("eeg_movement_classifier.keras")
print("Model saved as eeg_movement_classifier.keras")