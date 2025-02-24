import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data
df = pd.read_csv('cleaned_eeg_data.csv')

# Extract features and labels
eeg_channels = [col for col in df.columns if 'EEG' in col]
X = df[eeg_channels].values  # EEG signal data
y = df['label'].values       # Labels ('left' or 'right')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert 'left' -> 0, 'right' -> 1
y = to_categorical(y)  # Convert to one-hot encoding

# Reshape X to (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define CNN-LSTM model properly
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation='relu'),  # ✅ Removed return_sequences
    MaxPooling1D(pool_size=2),

    LSTM(50, return_sequences=False),  # ✅ Make sure LSTM gets 3D input
    Dropout(0.5),

    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 output classes: left and right
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# ✅ Define ModelCheckpoint BEFORE training
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min', 
    verbose=1
)

# ✅ Train the model
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[checkpoint]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.2f}')
