import os
from fastapi import FastAPI, HTTPException, Depends
import pandas as pd
import tensorflow as tf

import os
from fastapi import FastAPI, HTTPException, Depends
import pandas as pd
import tensorflow as tf

app = FastAPI(title="Epilepsy Detection API", 
              description="API for predicting epilepsy from patient data",
              version="1.0.0")

# Updated paths using environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/epilepsy_model.keras")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/patients/patients_data_updated.csv")

authenticated_users = {}

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅Model loaded successfully.")

print(f"Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded. Shape: {data.shape}, Columns: {data.columns.tolist()}")

# Dummy authentication system: just check if the ID and code match
def authenticate(patient_id: int, code: str):
    if code != "epilepsy":
        raise HTTPException(status_code=401, detail="Invalid authentication code")
    if patient_id < 1 or patient_id > len(data):
        raise HTTPException(status_code=404, detail="Patient ID not found")
    
    # If code is correct and ID is valid, we authenticate
    authenticated_users[patient_id] = True
    return patient_id

@app.get("/")
def read_root():
    return {"status": "online", "message": "Epilepsy Prediction API is running"}

@app.get("/predict/{patient_id}")
def predict(patient_id: int, code: str, authenticated_patient: int = Depends(authenticate)):
    """Get epilepsy prediction for a specific patient"""
    try:
        print(f"📡 Received prediction request for patient_id={patient_id}")
        
        # Get patient data
        row = data.iloc[patient_id - 1]
        print(f"🔎 Processing data for patient {patient_id}")
        
        # Prepare input data for model
        input_data = row.drop("ID", errors='ignore').values.reshape(1, 178, 1)
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_value = float(prediction[0][1])
        predicted_class = "Epilepsy" if prediction_value > 0.5 else "No Epilepsy"
        
        print(f"✅ Prediction for patient {patient_id}: {predicted_class} ({prediction_value:.4f})")
        
        return {
            "patient_id": patient_id,
            "prediction": predicted_class,
            "confidence": prediction_value if predicted_class == "Epilepsy" else 1 - prediction_value
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
