# Objectif: recharger données, (re)entraîner, sauvegarder scaler + modèle + metrics
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np, os, json

os.makedirs("models", exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(model,  "models/random_forest.pkl")
with open("models/metrics.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f)

print("saved models/* with metrics:", {"mse": mse, "r2": r2})
