import mlflow
import mlflow.keras
import json
import os
from tensorflow.keras.models import load_model

# Fonction pour loguer le modèle et les métriques dans MLflow
def track_model_and_metrics(model_path, metrics):
    # Commencer une session MLflow
    with mlflow.start_run():
        # Charger le modèle avec Keras
        model = load_model(model_path)  # Use Keras to load the model from local file

        
        # Loguer le modèle
        mlflow.keras.log_model(model, "model")
        
        # Loguer les métriques
        mlflow.log_metric("train_accuracy", metrics['train_accuracy'])
        mlflow.log_metric("val_accuracy", metrics['val_accuracy'])
        mlflow.log_metric("train_loss", metrics['train_loss'])
        mlflow.log_metric("val_loss", metrics['val_loss'])
        
        # Loguer les paramètres (epochs, batch_size, architecture, etc.)
        mlflow.log_param("epochs", metrics['epochs'])
        mlflow.log_param("batch_size", metrics['batch_size'])
        mlflow.log_param("architecture", metrics['architecture'])
    
    print("🧾 Model and metrics logged to MLflow successfully!")

if __name__ == "__main__":
    # Charger les informations sauvegardées dans 'models/metrics/model_metrics.json'
    with open("models/metrics/model_metrics.json", "r") as f:
        data = json.load(f)
    
    # Extraire le modèle path et les métriques
    model_path = data["model_path"]
    metrics = data["metrics"]
    
    # Loguer dans MLflow
    track_model_and_metrics(model_path, metrics)
