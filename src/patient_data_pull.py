import pandas as pd

PATIENT_DATA_PATH = "data/patients_data.csv"
OUTPUT_PATH = "data/patients_data_updated.csv"

df_patients = pd.read_csv(PATIENT_DATA_PATH)
df_patients = df_patients.iloc[:, :-1]
df_patients.insert(0, 'ID', range(1, len(df_patients) + 1))
df_patients.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Données des patients mises à jour : {OUTPUT_PATH}")
print(f"📊 Nombre de patients : {df_patients.shape[0]}")
print(f"📊 Nombre de features : {df_patients.shape[1]}")

