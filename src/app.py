from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import pandas as pd

MODEL_PATH = "models/epilepsy_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

PATIENT_DATA_PATH = "data/patients_data.csv"
df = pd.read_csv(PATIENT_DATA_PATH)

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
    X_patient = patient_data.iloc[:, 1:].values.reshape(1, 178, 1) 

    # Prédiction
    prediction = model.predict(X_patient)
    seizure = int(np.argmax(prediction))  
    
    return {"patient_id": patient_id, "seizure": seizure}
