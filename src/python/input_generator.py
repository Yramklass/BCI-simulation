import numpy as np
import pandas as pd

# Configuration
NUM_TIMESTEPS = 128
NUM_CHANNELS = 22  # Original EEG channels
NUM_FEATURES = NUM_CHANNELS * 2  # + delta features
OUTPUT_FILE = "../cpp/eeg_data.csv"

# Generate synthetic data
np.random.seed(42)

# Base signal (simulating left/right movement patterns)
def generate_movement_signal(timesteps, is_right_movement):
    base_signal = np.sin(np.linspace(0, 10, timesteps))
    if is_right_movement:
        # Stronger activity in left hemisphere (C3 channels)
        c3_pattern = 0.5 * np.sin(np.linspace(0, 5, timesteps))
        c4_pattern = 0.2 * np.sin(np.linspace(0, 3, timesteps))
    else:
        # Stronger activity in right hemisphere (C4 channels)
        c3_pattern = 0.2 * np.sin(np.linspace(0, 3, timesteps))
        c4_pattern = 0.5 * np.sin(np.linspace(0, 5, timesteps))
    
    return base_signal, c3_pattern, c4_pattern

# Create a sample (change to True for right movement)
base, c3, c4 = generate_movement_signal(NUM_TIMESTEPS, is_right_movement=False)

# Generate all channels
data = []
for t in range(NUM_TIMESTEPS):
    # Original channels (1-22)
    row = []
    
    # Simulate EEG channels (including C3 at index 0, C4 at index 2 in your channel order)
    for ch in range(NUM_CHANNELS):
        if ch == 0:  # C3
            val = base[t] + c3[t] + np.random.normal(0, 0.1)
        elif ch == 2:  # C4
            val = base[t] + c4[t] + np.random.normal(0, 0.1)
        else:
            val = base[t] + np.random.normal(0, 0.2)
        row.append(val)
    
    # Delta features (23-44) - simulated as differences
    for ch in range(NUM_CHANNELS):
        if t > 0:
            delta = row[ch] - data[t-1][ch]
        else:
            delta = 0.0
        row.append(delta)
    
    data.append(row)

# Save to CSV
pd.DataFrame(data).to_csv(OUTPUT_FILE, index=False, header=False)
print(f"Generated test file: {OUTPUT_FILE}")