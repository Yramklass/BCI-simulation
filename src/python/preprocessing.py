import pandas as pd
import numpy as np
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pickle
import datetime

def preprocess_eeg_data(filename, window_size=128, overlap=0.5, sampling_rate=250):
    """
    Preprocess EEG data with overlapping windows and feature extraction
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file containing EEG data
    window_size : int
        Size of the sliding window in samples
    overlap : float
        Overlap ratio between consecutive windows (0-1)
    sampling_rate : int
        Sampling rate of the EEG data in Hz
    
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed features
    y : numpy.ndarray
        Labels
    """
    print(f"Loading data from {filename}...")
    # Load dataset
    df = pd.read_csv(filename)
    
    # Filter out only left and right labels
    df = df[df['label'].isin(['left', 'right'])]
    print(f"Found {len(df)} samples with left/right labels")
    
    # Sort by patient and time to ensure correct ordering
    df = df.sort_values(by=["patient", "time"])
    
    # Extract EEG channel names
    eeg_columns = df.columns[4:]
    print(f"Found {len(eeg_columns)} EEG channels")
    
    # Calculate stride for overlapping windows
    stride = int(window_size * (1 - overlap))
    
    # Store segmented data
    segmented_data = []
    segmented_labels = []
    patient_ids = []
    window_info = []  # Store information about each window for traceability
    
    # Group by patient (to avoid mixing different recordings)
    print("Segmenting data into windows...")
    for patient_id, patient_data in df.groupby("patient"):
        patient_data = patient_data.reset_index(drop=True)
        
        # Iterate over windows with overlap
        for start_idx in range(0, len(patient_data) - window_size + 1, stride):
            window = patient_data.iloc[start_idx : start_idx + window_size]
            
            # Extract EEG signals
            eeg_values = window[eeg_columns].values
            
            # Determine label for this window (most frequent label)
            window_label = window["label"].mode()[0]
            
            # Only include windows with consistent labels (>90% same label)
            label_counts = window["label"].value_counts(normalize=True)
            if label_counts.iloc[0] > 0.9:
                segmented_data.append(eeg_values)
                segmented_labels.append(window_label)
                patient_ids.append(patient_id)
                
                # Store window metadata
                window_info.append({
                    'patient_id': patient_id,
                    'start_time': window['time'].iloc[0],
                    'end_time': window['time'].iloc[-1],
                    'label': window_label,
                    'window_start_idx': start_idx,
                    'window_end_idx': start_idx + window_size - 1
                })
    
    print(f"Created {len(segmented_data)} windows from {len(df.groupby('patient'))} patients")
    
    # Convert lists to numpy arrays
    X = np.array(segmented_data)  # Shape: (num_windows, window_size, num_channels)
    y = np.array(segmented_labels)  # Shape: (num_windows,)
    
    print("Extracting features...")
    # Extract time and frequency domain features
    X_features = extract_features(X, sampling_rate)
    
    # Normalize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # Convert labels to binary (left=0, right=1)
    y_binary = np.array([0 if label == 'left' else 1 for label in y])
    
    # Create DataFrame with window info
    window_info_df = pd.DataFrame(window_info)
    
    # Create DataFrame with features
    feature_df = pd.DataFrame(X_features)
    
    # Create DataFrames to save
    window_df = pd.concat([window_info_df, feature_df], axis=1)
    
    # Also save the scaler for future use
    scaler_info = {
        'scaler': scaler,
        'feature_names': [f'feature_{i}' for i in range(X_features.shape[1])]
    }
    
    return X_features, y_binary, np.array(patient_ids), window_df, scaler_info

def extract_features(X, sampling_rate):
    """
    Extract time and frequency domain features from EEG windows
    
    Parameters:
    -----------
    X : numpy.ndarray
        EEG data of shape (num_windows, window_size, num_channels)
    sampling_rate : int
        Sampling rate of the EEG data in Hz
    
    Returns:
    --------
    features : numpy.ndarray
        Extracted features of shape (num_windows, num_features)
    """
    num_windows, window_size, num_channels = X.shape
    features = []
    
    # Frequency bands (Hz)
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    for i in range(num_windows):
        window_features = []
        
        for ch in range(num_channels):
            channel_data = X[i, :, ch]
            
            # Time domain features
            window_features.append(np.mean(channel_data))
            window_features.append(np.std(channel_data))
            window_features.append(np.max(channel_data) - np.min(channel_data))
            
            # Frequency domain features
            freqs, psd = signal.welch(channel_data, fs=sampling_rate, nperseg=min(window_size, 256))
            
            # Band power features
            for band_name, (low, high) in bands.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.mean(psd[idx_band])
                window_features.append(band_power)
        
        features.append(window_features)
    
    return np.array(features)

def save_preprocessed_data(X, y, patient_ids, window_df, scaler_info, output_dir):
    """
    Save preprocessed data to files
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    patient_ids : numpy.ndarray
        Patient IDs
    window_df : pandas.DataFrame
        Window information and features
    scaler_info : dict
        Scaler information
    output_dir : str
        Directory to save output files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save window DataFrame
    window_df.to_csv(os.path.join(output_dir, "preprocessed_windows.csv"), index=False)
    print(f"Preprocessed window data saved to {os.path.join(output_dir, 'preprocessed_windows.csv')}")
    
    # Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler_info, f)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, "X_features.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    np.save(os.path.join(output_dir, "patient_ids.npy"), patient_ids)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEG Data Preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="./preprocessed_data", help="Output directory")
    parser.add_argument("--window", type=int, default=128, help="Window size in samples")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio (0-1)")
    parser.add_argument("--sampling-rate", type=int, default=250, help="EEG sampling rate in Hz")
    
    args = parser.parse_args()
    
    # Preprocess data
    X, y, patient_ids, window_df, scaler_info = preprocess_eeg_data(
        args.input, args.window, args.overlap, args.sampling_rate
    )
    
    # Save preprocessed data
    save_preprocessed_data(X, y, patient_ids, window_df, scaler_info, args.output)