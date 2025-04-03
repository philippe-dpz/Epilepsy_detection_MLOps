import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

RAW_DATA_PATH = "data/raw/Epileptic_Seizure_Recognition.csv"
PROCESSED_X_TRAIN_PATH = "data/processed/X_train.npy"
PROCESSED_Y_TRAIN_PATH = "data/processed/Y_train.npy"
PROCESSED_X_TEST_PATH = "data/processed/X_test.npy"
PROCESSED_Y_TEST_PATH = "data/processed/Y_test.npy"
PATIENT_DATA_PATH = "data/patients_data.csv"

df = pd.read_csv(RAW_DATA_PATH)

df = df.iloc[:, 1:]

# Sélectionner aléatoirement 800 lignes pour l'entraînement
df_train = df.sample(n=800, random_state=42)

# Séparer features (X) et labels (y) pour l'entraînement
X_train_full = df_train.iloc[:, :-1].values  
y_train_full = df_train.iloc[:, -1].values   

# Transformer les labels en binaire 
y_train_full = np.where(y_train_full == 1, 1, 0)

Y_train_full = to_categorical(y_train_full, num_classes=2)

# Split train/test (80% train, 20% test) 
X_train, X_test, Y_train, Y_test = train_test_split(X_train_full, Y_train_full, test_size=0.20, random_state=42)

# Reshape pour le modèle LSTM (ajout d'une dimension pour la compatibilité)
X_train = X_train.reshape(-1, 178, 1)
X_test = X_test.reshape(-1, 178, 1)


# Sauvegarder les données traitées
np.save(PROCESSED_X_TRAIN_PATH, X_train)
np.save(PROCESSED_Y_TRAIN_PATH, Y_train)
np.save(PROCESSED_X_TEST_PATH, X_test)
np.save(PROCESSED_Y_TEST_PATH, Y_test)

df_remaining = df.drop(df_train.index)
df_remaining.to_csv(PATIENT_DATA_PATH, index=False)

# Afficher les shapes pour vérification
print(f"✅ Preprocessing terminé !")
print(f"📊 X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"📊 X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
print(f"📊 Patient dataset saved at {PATIENT_DATA_PATH}")
