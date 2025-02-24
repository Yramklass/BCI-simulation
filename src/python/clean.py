import pandas as pd
import numpy as np
from scipy.stats import mode

def segment_eeg_data(filename, window_size=128, sampling_rate=250):
    # Load dataset
    df = pd.read_csv(filename)
    
    # Sort by patient and time to ensure correct ordering
    df = df.sort_values(by=["patient", "time"])
    
    # Extract EEG channel names (assuming they start from column 4 onward)
    eeg_columns = df.columns[4:]
    
    # Store segmented data
    segmented_data = []
    segmented_labels = []
    
    # Group by patient (to avoid mixing different recordings)
    for patient_id, patient_data in df.groupby("patient"):
        patient_data = patient_data.reset_index(drop=True)
        
        # Iterate over windows
        for start_idx in range(0, len(patient_data) - window_size, window_size):
            window = patient_data.iloc[start_idx : start_idx + window_size]
            
            # Extract EEG signals
            eeg_values = window[eeg_columns].values
            
            # Determine label for this window (most frequent label)
            window_label = mode(window["label"], keepdims=True).mode[0]
            
            segmented_data.append(eeg_values)
            segmented_labels.append(window_label)
    
    # Convert lists to numpy arrays
    segmented_data = np.array(segmented_data)  # Shape: (num_windows, 128, num_channels)
    segmented_labels = np.array(segmented_labels)  # Shape: (num_windows,)
    
    return segmented_data, segmented_labels

# Example usage
# eeg_data, labels = segment_eeg_data("eeg_data.csv")
# print("Segmented EEG Shape:", eeg_data.shape)  # (num_windows, 128, num_channels)
# print("Labels Shape:", labels.shape)  # (num_windows,)
