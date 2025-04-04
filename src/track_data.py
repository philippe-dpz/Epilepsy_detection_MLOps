import subprocess

def track_data():
    # Ajouter les fichiers traités à DVC
    subprocess.run(['dvc', 'add', 'data/processed/X_train.npy'])
    subprocess.run(['dvc', 'add', 'data/processed/Y_train.npy'])
    subprocess.run(['dvc', 'add', 'data/processed/X_test.npy'])
    subprocess.run(['dvc', 'add', 'data/processed/Y_test.npy'])
    
    # Ajouter les fichiers .dvc à Git
    subprocess.run(['git', 'add', 'data/processed/*.dvc'])
    subprocess.run(['git', 'commit', '-m', 'Ajout des fichiers de données'])
    
    # Optionnel : Pousser les fichiers vers le dépôt DVC distant
    subprocess.run(['dvc', 'push'])

if __name__ == "__main__":
    track_data()
