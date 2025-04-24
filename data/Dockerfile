# Base Python officielle légère
FROM python:3.9.21-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances Linux (optionnelles)
RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copier l’ensemble du projet dans le conteneur
COPY . .

# Exposer le port de FastAPI
EXPOSE 8000

# Lancer ton API automatiquement au démarrage du conteneur
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
