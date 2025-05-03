import pandas as pd
import numpy as np

# Parameters
n_epochs = 2            # Number of fake recording epochs
samples_per_epoch = 256  # Must be >=128 to get at least one window
n_patients = 1
eeg_channels = [
    'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fz', 'EEG-Pz',
    'EEG-00','EEG-01', 'EEG-02', 'EEG-03', 'EEG-04', 'EEG-05',
    'EEG-06', 'EEG-07', 'EEG-08', 'EEG-09', 'EEG-10',
    'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15',
    'EEG-16'
]
#patient,time,label,epoch,EEG-Fz,EEG-0,EEG-1,EEG-2,EEG-3,EEG-4,EEG-5,EEG-C3,EEG-6,EEG-Cz,EEG-7,EEG-C4,EEG-8,EEG-9,EEG-10,EEG-11,EEG-12,EEG-13,EEG-14,EEG-Pz,EEG-15,EEG-16

# Generate fake EEG data
rows = []
for patient_id in range(n_patients):
    for epoch in range(n_epochs):
        for t in range(samples_per_epoch):
            row = {
                'patient': f'P{patient_id}',
                'epoch': epoch,
                'time': t,
                'label': np.random.choice(['left', 'right']),  # Optional
            }
            for ch in eeg_channels:
                row[ch] = np.random.normal(loc=0, scale=1)  # Simulated EEG value
            rows.append(row)

# Create DataFrame and save
df = pd.DataFrame(rows)
df.to_csv("test_input.csv", index=False)

print("Test input file 'test_input.csv' generated.")
