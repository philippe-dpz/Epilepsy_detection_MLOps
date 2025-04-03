import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam

# Définition des chemins des fichiers
PROCESSED_X_TRAIN_PATH = "data/processed/X_train.npy"
PROCESSED_Y_TRAIN_PATH = "data/processed/Y_train.npy"
PROCESSED_X_TEST_PATH = "data/processed/X_test.npy"
PROCESSED_Y_TEST_PATH = "data/processed/Y_test.npy"
MODEL_SAVE_PATH = "models/epilepsy_model.h5"

X_train = np.load(PROCESSED_X_TRAIN_PATH)
Y_train = np.load(PROCESSED_Y_TRAIN_PATH)
X_test = np.load(PROCESSED_X_TEST_PATH)
Y_test = np.load(PROCESSED_Y_TEST_PATH)

# Afficher les dimensions des datasets
print(f"📊 X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"📊 X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Définition du modèle
model = Sequential()
model.add(LSTM(56, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(56))
model.add(Dropout(0.3))
model.add(Dense(20, activation='tanh'))
model.add(Dense(2, activation='softmax')) 

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Entraînement du modèle avec validation
hist = model.fit(
    X_train, Y_train,
    epochs=56,
    batch_size=15,
    validation_data=(X_test, Y_test),  
    shuffle=False
)

# Sauvegarde du modèle entraîné
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)

print(f"✅ Entraînement terminé ! Modèle sauvegardé à {MODEL_SAVE_PATH}")
