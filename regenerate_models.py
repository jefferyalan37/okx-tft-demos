import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create models directory
os.makedirs("models", exist_ok=True)

# 1. Train and Save Ensemble Model (Random Forest Regressor)
print("Training Ensemble Model...")
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ensemble_model = RandomForestRegressor(n_estimators=100, random_state=42)
ensemble_model.fit(X_train, y_train)

# Save the trained ensemble model
with open("models/ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)
print("Ensemble Model Saved: models/ensemble_model.pkl")

# 2. Train and Save TFT Model (PyTorch)
print("Training Temporal Fusion Transformer Model...")
class SimpleTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleTFT, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Define model, loss, and optimizer
input_dim = 10
hidden_dim = 64
tft_model = SimpleTFT(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(tft_model.parameters(), lr=0.001)

# Generate synthetic data
X_tft = torch.tensor(X_train, dtype=torch.float32)
y_tft = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Train the model
epochs = 100
for epoch in range(epochs):
    tft_model.train()
    optimizer.zero_grad()
    outputs = tft_model(X_tft)
    loss = criterion(outputs, y_tft)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained TFT model
torch.save(tft_model.state_dict(), "models/tft_model.pt")
print("TFT Model Saved: models/tft_model.pt")
