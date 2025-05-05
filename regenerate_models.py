# regenerate_models.py
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import os

os.makedirs("models", exist_ok=True)

# Dummy data
X = np.array([[i] for i in range(10)])
y = np.array([i * 2 for i in range(10)])

# Scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "models/scaler.pkl")

# XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
xgb_model.save_model("models/xgb.json")

# Stacking model
stack_model = RandomForestRegressor()
stack_model.fit(X, y)
joblib.dump(stack_model, "models/stack.pkl")

# Fake TFT state dict
tft_state_dict = {"weights": [1, 2, 3]}
joblib.dump(tft_state_dict, "models/tft_state_dict.pt")

print("✅ Model artifacts successfully regenerated.")
