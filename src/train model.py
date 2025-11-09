from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

X_train_scaled = joblib.load("models/X_train_scaled.pkl")
y_train = joblib.load("models/y_train.pkl")

# CréATION ET ENTRAINEMENT DU MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Sauvegarde du modèle
joblib.dump(model, "models/random_forest.pkl")