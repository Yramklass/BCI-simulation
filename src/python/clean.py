import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "BCICIV_2a_all_patients.csv"  
df = pd.read_csv(file_path)

# Filter only 'left' and 'right' labels
df = df[df["label"].isin(["left", "right"])]

# Drop irrelevant columns
df = df.drop(columns=["patient", "time", "epoch"])

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Normalize EEG features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Save cleaned data
cleaned_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
cleaned_data.to_csv("cleaned_dataset.csv", index=False)

print("Preprocessing complete. Cleaned data saved as 'cleaned_dataset.csv'.")
