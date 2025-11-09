from sklearn.metrics import mean_squared_error, r2_score
import joblib

model = joblib.load("models/random_forest.pkl")
X_test_scaled = joblib.load("models/X_test_scaled.pkl")
y_test = joblib.load("models/y_test.pkl")

y_pred = model.predict(X_test_scaled)

print("MSE :", mean_squared_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))
