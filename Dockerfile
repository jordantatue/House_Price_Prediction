FROM python:3.11-slim

# Installer curl pour le healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# 1) Créer un user non-root
RUN useradd -m appuser

# 2) Dossier et dépendances système minimales
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# 3) Copier requirements et installer
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copier le code et les modèles
COPY . /app

# 5) Droits & passage en non-root
RUN chown -R appuser:appuser /app
USER appuser

# 6) Exposer et démarrer
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
