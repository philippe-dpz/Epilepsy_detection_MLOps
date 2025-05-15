import os
import pandas as pd

# Get paths from environment variables with fallbacks
BASE_DATA_PATH = os.getenv("DATA_PATH", "/app/data")
PATIENT_DATA_PATH = os.path.join(BASE_DATA_PATH, "patients/patients_data.csv")
OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "patients/patients_data_updated.csv")

print(f"📂 Reading patient data from: {PATIENT_DATA_PATH}")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Read the patient data
df_patients = pd.read_csv(PATIENT_DATA_PATH)
df_patients = df_patients.iloc[:, :-1]

# Add ID column
df_patients.insert(0, 'ID', range(1, len(df_patients) + 1))

# Save the processed patient data
df_patients.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Patient data updated and saved to: {OUTPUT_PATH}")
print(f"📊 Number of patients: {df_patients.shape[0]}")
print(f"📊 Number of features: {df_patients.shape[1]}")