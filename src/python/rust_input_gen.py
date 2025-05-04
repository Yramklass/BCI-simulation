import numpy as np
import pandas as pd


NUM_TIMESTEPS = 128
NUM_CHANNELS = 22  # Original EEG channels
NUM_FEATURES = NUM_CHANNELS * 2  # Base channels + delta features
OUTPUT_FILE = "eeg_data.csv" 

# Choose which movement pattern to simulate
# False for simulated "left" movement (stronger C4/right hemisphere)
# True for simulated "right" movement (stronger C3/left hemisphere)
SIMULATE_RIGHT_MOVEMENT = True 

# Data Generation\
np.random.seed(42) # Keep seed for reproducibility

# Base signal and movement-specific patterns
def generate_movement_signal(timesteps, is_right_movement):
    base_signal = np.sin(np.linspace(0, 10, timesteps))
    if is_right_movement:
        # Stronger activity in left hemisphere (C3 channels assumed index 0)
        print("Generating data simulating RIGHT movement (stronger C3 signal)")
        c3_pattern = 0.8 * np.sin(np.linspace(0, 5, timesteps)) # Increased strength
        c4_pattern = 0.2 * np.sin(np.linspace(0, 3, timesteps))
    else:
        # Stronger activity in right hemisphere (C4 channels assumed index 2)
        print("Generating data simulating LEFT movement (stronger C4 signal)")
        c3_pattern = 0.2 * np.sin(np.linspace(0, 3, timesteps))
        c4_pattern = 0.8 * np.sin(np.linspace(0, 5, timesteps)) # Increased strength

    return base_signal, c3_pattern, c4_pattern

# Create the specific signal patterns
base, c3, c4 = generate_movement_signal(NUM_TIMESTEPS, is_right_movement=SIMULATE_RIGHT_MOVEMENT)

# Generate all channels and calculate deltas iteratively
data = []
for t in range(NUM_TIMESTEPS):
    row = [] # Holds NUM_FEATURES (44) values for this timestep

    # Generate Original Channels (0-21)
    raw_channels_this_step = []
    for ch in range(NUM_CHANNELS):
        noise = np.random.normal(0, 0.15) # Adjusted noise slightly
        C3_INDEX = 7
        C4_INDEX = 11

        if ch == C3_INDEX:
            val = base[t] + c3[t] + noise
        elif ch == C4_INDEX:
            val = base[t] + c4[t] + noise

        else: # Other channels
            val = base[t] + np.random.normal(0, 0.2) + noise # Added base noise too
        raw_channels_this_step.append(val)
    row.extend(raw_channels_this_step) # Add the 22 raw channels

    # Calculate Delta Features (22-43)
    delta_channels_this_step = []
    if t > 0:
        # Get the raw channels from the *previous* timestep (first 22 elements of data[t-1])
        previous_raw_channels = data[t-1][:NUM_CHANNELS]
        for ch in range(NUM_CHANNELS):
            delta = raw_channels_this_step[ch] - previous_raw_channels[ch]
            delta_channels_this_step.append(delta)
    else:
        # For the first timestep (t=0), deltas are zero
        delta_channels_this_step = [0.0] * NUM_CHANNELS
    row.extend(delta_channels_this_step) # Add the 22 delta channels

    # Append the full row (44 features) for this timestep
    data.append(row)

# Normalization

# Convert the list of lists to a NumPy array (float32 is standard for ML)
data_np = np.array(data, dtype=np.float32)
print(f"Generated data shape (before normalization): {data_np.shape}") # Should be (128, 44)

# Calculate mean and standard deviation for each FEATURE COLUMN (axis=0)
mean_per_feature = np.mean(data_np, axis=0)
std_per_feature = np.std(data_np, axis=0)

# Apply Z-score normalization: (value - mean) / (std + epsilon)
# Add a small epsilon to prevent division by zero if std happens to be 0
epsilon = 1e-8
normalized_data = (data_np - mean_per_feature) / (std_per_feature + epsilon)
print(f"Normalized data shape: {normalized_data.shape}")


pd.DataFrame(normalized_data).to_csv(OUTPUT_FILE, index=False, header=False)
print(f"Generated and NORMALIZED test file: {OUTPUT_FILE}")