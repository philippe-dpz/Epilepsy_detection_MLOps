import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json

# Define file paths
PROCESSED_X_TRAIN_PATH = "data/processed/X_train.npy"
PROCESSED_Y_TRAIN_PATH = "data/processed/Y_train.npy"
PROCESSED_X_TEST_PATH = "data/processed/X_test.npy"
PROCESSED_Y_TEST_PATH = "data/processed/Y_test.npy"
MODEL_SAVE_PATH = "models/epilepsy_model.h5"
METRICS_SAVE_PATH = "models/metrics/model_metrics.json"  # Corrected path for metrics file

# Load the data
X_train = np.load(PROCESSED_X_TRAIN_PATH)
Y_train = np.load(PROCESSED_Y_TRAIN_PATH)
X_test = np.load(PROCESSED_X_TEST_PATH)
Y_test = np.load(PROCESSED_Y_TEST_PATH)

# Define the model
model = Sequential([
    LSTM(56, input_shape=(X_train.shape[1], 1), return_sequences=True),
    Dropout(0.3),
    LSTM(56),
    Dropout(0.3),
    Dense(20, activation='tanh'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=5,         # Patience of 5 epochs
                               restore_best_weights=True)  # Restore the model with the best weights

# Train the model
hist = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=15,
    validation_data=(X_test, Y_test),  # Use X_test and Y_test as validation data
    shuffle=False,
    callbacks=[early_stopping]  # Add early stopping callback here
)

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)  # Ensure the directory exists
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at {MODEL_SAVE_PATH}")

# Collect training metrics
metrics = {
    "train_accuracy": float(hist.history["accuracy"][-1]),
    "val_accuracy": float(hist.history["val_accuracy"][-1]),
    "train_loss": float(hist.history["loss"][-1]),
    "val_loss": float(hist.history["val_loss"][-1]),
    "epochs": 20,
    "batch_size": 15,
    "architecture": "2xLSTM_+_Dense"
}

# Save the model path and metrics for tracking
os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)  # Ensure the metrics directory exists
with open(METRICS_SAVE_PATH, "w") as f:
    json.dump({"model_path": MODEL_SAVE_PATH, "metrics": metrics}, f)

print(f"✅ Training complete. Metrics saved in '{METRICS_SAVE_PATH}'")
