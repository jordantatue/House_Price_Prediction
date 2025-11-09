# Image Python officielle, légère
FROM python:3.11-slim AS base

# — Bonnes pratiques Python —
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# — Dépendances système minimales (si besoin) —
# (ajoute gcc, libgomp, etc. si certaines libs scientifiques le réclament)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# 1) Copier uniquement requirements pour profiter du cache pip
COPY requirements.txt /app/requirements.txt

# 2) Installer dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copier le reste du projet (code + modèles LFS déjà récupérés par la CI)
COPY . /app

# — Optionnel : vérification que les modèles sont bien là —
# RUN ls -la /app/models || true

# Expose l’API FastAPI
EXPOSE 8000

# Commande de démarrage (Uvicorn)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
