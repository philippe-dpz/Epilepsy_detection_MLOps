from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import pandas as pd

# Charger le modèle
MODEL_PATH = "models/epilepsy_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Charger les données des patients
PATIENT_DATA_PATH = "data/patients_data.csv"
df = pd.read_csv(PATIENT_DATA_PATH)

# Init API
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API d'épilepsie"}

@app.get("/predict/{patient_id}")
def predict_seizure(patient_id: int):
    # Récupérer les données du patient en fonction de son ID
    patient_data = df[df["ID"] == patient_id]
    
    if patient_data.empty:
        return {"error": "Patient non trouvé"}
    
    # Préparer les données pour la prédiction
    X_patient = patient_data.iloc[:, 1:].values.reshape(1, 178, 1)  # Assurer la bonne forme (178 time points, 1 feature per time point)

    # Prédiction
    prediction = model.predict(X_patient)
    seizure = int(np.argmax(prediction))  # 1 = crise, 0 = pas de crise
    
    return {"patient_id": patient_id, "seizure": seizure}
