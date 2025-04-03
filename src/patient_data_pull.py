import pandas as pd

PATIENT_DATA_PATH = "data/patients_data.csv"


# Charger le fichier CSV des patients
df_patients = pd.read_csv(PATIENT_DATA_PATH)

# Supprimer la colonne des labels (si elle existe)
df_patients = df_patients.iloc[:, :-1] # Utilisation de 'errors="ignore"' pour ne pas avoir d'erreur si la colonne "label" est absente

# Ajouter une colonne 'ID' au début du dataset
df_patients.insert(0, 'ID', range(1, len(df_patients) + 1))

# Sauvegarder le nouveau dataset avec les ID
df_patients.to_csv(PROCESSED_PATIENT_DATA_PATH, index=False)

print(f"✅ Données des patients mises à jour : {PATIENT_DATA_PATH}")
print(f"📊 Nombre de patients : {df_patients.shape[0]}")
print(f"📊 Nombre de features : {df_patients.shape[1]}")

