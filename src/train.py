import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mlflow
from .preprocessing import preprocess_data

def build_model(input_shape: tuple) -> Sequential:
    """Define LSTM model architecture."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='tanh'),
        Dense(5, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(data_path: str, epochs: int = 50):
    """End-to-end training pipeline with MLflow tracking."""

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    mlflow.set_experiment("epilepsy_classification")
    
    with mlflow.start_run():
        
        mlflow.log_param("epochs", epochs)
        
        model = build_model((X_train.shape[1], X_train.shape[2]))
        history = model.fit(X_train, y_train, 
                          validation_data=(X_test, y_test),
                          epochs=epochs, 
                          batch_size=32)
        
        
        mlflow.log_metrics({
            "train_accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1]
        })
        
       
        model.save("models/epilepsy_model.h5")
        mlflow.log_artifact("models/epilepsy_model.h5")