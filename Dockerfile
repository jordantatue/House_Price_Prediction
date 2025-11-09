# Utilise une image Python officielle légère
FROM python:3.14-slim

# Crée un dossier dans le conteneur pour ton app
WORKDIR /app

# Copie les fichiers de ton projet dans le conteneur
COPY . /app

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port du serveur FastAPI
EXPOSE 8000

# Commande de démarrage du serveur
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
