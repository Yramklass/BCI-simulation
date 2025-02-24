import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=250, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def process_eeg_data(file_path, window_size=500):
    df = pd.read_csv(file_path)
    
    # Filter only 'left' and 'right' labels
    df = df[df['label'].isin(['left', 'right'])]

    # Normalization (Z-score per channel)
    eeg_channels = [col for col in df.columns if 'EEG' in col]
    scaler = StandardScaler()
    df[eeg_channels] = scaler.fit_transform(df[eeg_channels])
    
    # Apply bandpass filter
    df[eeg_channels] = bandpass_filter(df[eeg_channels].values)
    
    # Reshape into 500ms windows (assuming 250Hz sampling rate -> 125 samples per window)
    fs = 250  # Sample rate in Hz
    samples_per_window = int((window_size / 1000) * fs)
    
    segmented_data = []
    for i in range(0, len(df) - samples_per_window + 1, samples_per_window):
        segment = df.iloc[i:i+samples_per_window]
        label = segment['label'].mode().values[0] if not segment['label'].mode().empty else 'unknown'
        epoch = segment['epoch'].mode().values[0] if not segment['epoch'].mode().empty else -1
        segment_mean = segment[eeg_channels].mean().values  # Aggregate features
        segmented_data.append([label, epoch] + list(segment_mean))
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(segmented_data, columns=['label', 'epoch'] + eeg_channels)
    
    # Check if DataFrame is empty before saving
    if processed_df.empty:
        print("Warning: Processed data is empty! Check input CSV.")
    else:
        processed_df.to_csv('cleaned_eeg_data.csv', index=False)
        print("Preprocessing complete. Saved as cleaned_eeg_data.csv")
    
    return processed_df
    

# Example usage:
# processed_data = process_eeg_data('your_eeg_data.csv')

if __name__ == "__main__":
    processed_data = process_eeg_data('BCICIV_2a_all_patients.csv')  # Replace with actual file name
