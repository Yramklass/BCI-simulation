import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report

# Load and preprocess data
df = pd.read_csv("BCICIV_2a_all_patients.csv")
df = df[df['label'].isin(['left', 'right'])]
df['label'] = df['label'].map({'left': 0, 'right': 1})

# Channel ordering and normalization
channel_order = ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz'] + \
                [col for col in df.columns if col.startswith("EEG") and col not in ['EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz']]
eeg_channels = [col for col in channel_order if col in df.columns]

for patient in df['patient'].unique():
    patient_mask = df['patient'] == patient
    for channel in eeg_channels:
        df.loc[patient_mask, channel] = (df.loc[patient_mask, channel] - 
                                        df.loc[patient_mask, channel].mean()) / \
                                       (df.loc[patient_mask, channel].std() + 1e-8)

# Windowing function with delta features
def create_windows(dataframe, window_size=128, step_size=32):
    windows, labels = [], []
    for _, epoch_df in dataframe.groupby("epoch"):
        signals = epoch_df[eeg_channels].values
        signals = np.concatenate([signals, np.diff(signals, axis=0, prepend=signals[0:1])], axis=1)
        epoch_labels = epoch_df["label"].values
        for start in range(0, len(signals) - window_size + 1, step_size):
            windows.append(signals[start:start+window_size])
            labels.append(np.bincount(epoch_labels[start:start+window_size]).argmax())
    return np.array(windows), np.array(labels)

# Patient-wise split
test_patient = df['patient'].unique()[-1]
X_train, y_train = create_windows(df[df['patient'] != test_patient])
X_test, y_test = create_windows(df[df['patient'] == test_patient])


# Build model with MHSA
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Temporal convolutional block
    x = layers.Conv1D(64, 5, padding='same', activation='swish')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Channel attention (SE-style)
    channel_att = layers.GlobalAvgPool1D()(x)
    channel_att = layers.Dense(64 // 4, activation='relu')(channel_att)
    channel_att = layers.Dense(64, activation='sigmoid')(channel_att)
    channel_att = layers.Reshape((1, 64))(channel_att)
    x = layers.Multiply()([x, channel_att])
    
    # Deeper conv block
    x = layers.Conv1D(128, 3, padding='same', activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # MHSA block
    mha_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, mha_out])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAvgPool1D()(x)

    # Classification head
    x = layers.Dense(32, activation='swish')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)

model = build_transformer_model(X_train.shape[1:])

# Improved learning schedule
initial_lr = 1e-4
lr_schedule = optimizers.schedules.CosineDecayRestarts(
    initial_lr,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9
)

# Focal loss for class imbalance
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + 1e-7))

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss,
    metrics=["accuracy", tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks_list = [
    callbacks.ModelCheckpoint("best_transformer_model.keras", monitor="val_auc", mode='max', save_best_only=True),
    callbacks.EarlyStopping(monitor="val_auc", patience=20, mode='max', restore_best_weights=True)
]

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=200,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nEnhanced Evaluation Report:")
print(classification_report(y_test, y_pred, digits=4))