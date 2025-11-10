import joblib, json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/random_forest.pkl")

X_test_s = scaler.transform(X_test)
pred = model.predict(X_test_s)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print({"mse": mse, "r2": r2})
with open("models/metrics.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f)
