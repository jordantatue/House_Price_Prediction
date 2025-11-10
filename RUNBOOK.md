# RUNBOOK - Housing API

## Symptôme: /health DOWN
1. `docker compose ps` — vérifier conteneurs.
2. `docker logs ml_api` — lire erreurs FastAPI/uvicorn.
3. `curl http://api:8000/metrics` — si KO, l’API ne démarre pas (modèle absent ?).
4. Vérifier présence `models/random_forest.pkl` & `scaler.pkl`.

## Symptôme: Pas de métriques dans Prometheus
1. `curl http://api:8000/metrics` -> doit renvoyer du texte.
2. `curl http://prometheus:9090/-/healthy` -> OK si healthy.
3. Inspecter `prometheus.yml` (cible `api:8000`) et `docker compose logs prometheus`.

## Symptôme: Alertes
- `APIDown` : vérifier `api` et réseau.
- `NoPredictions` : générer des prédictions (`curl POST /predict`) ou revoir trafic.
