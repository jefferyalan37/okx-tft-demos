import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
BASE_MODEL_DIR = os.getenv("MODEL_DIR", "./models/")
MODEL_PATH = os.path.join(BASE_MODEL_DIR, "tft_model.pkl")
SCALER_PATH = os.path.join(BASE_MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_MODEL_DIR, "required_features.pkl")
DAILY_DATA_PATH = "./OKXDemoProject-5/data/final_bitcoin_daily_processed.csv"
HOURLY_DATA_PATH = "./OKXDemoProject-5/data/final_bitcoin_hourly_processed.csv"
TFT_PATH = os.path.join(BASE_MODEL_DIR, "tft_state_dict.pt")
torch.save(model.state_dict(), TFT_PATH)
model = TemporalFusionTransformerComplete(...)
model.load_state_dict(torch.load(TFT_PATH))
model.eval()
############################################
#       Indicator Functions
############################################
def compute_sma(df, window):
    return df['close'].rolling(window=window).mean()


def compute_ema(df, window):
    return df['close'].ewm(span=window, adjust=False).mean()

def compute_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_bollinger_bands(df, window=20, num_std=2):
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, lower_band

def compute_macd(df, short_win=12, long_win=26, signal_win=9):
    ema_short = df['close'].ewm(span=short_win, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_win, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_win, adjust=False).mean()
    return macd, signal
############################################
#       Preprocessing Function
############################################
def preprocess_data(data):
    print("[DEBUG] Initial shape:", data.shape)
    print("[DEBUG] Columns:", data.columns.tolist())

    # Ensure required columns exist
    if 'close' not in data.columns or 'volume' not in data.columns:
        print("[ERROR] DataFrame missing 'close' or 'volume' column.")
        return pd.DataFrame(), None

    data['SMA_10'] = compute_sma(data, 10)
    data['SMA_20'] = compute_sma(data, 20)
    data['EMA_10'] = compute_ema(data, 10)
    data['EMA_20'] = compute_ema(data, 20)
    data['RSI_14'] = compute_rsi(data, 14)
    data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data, 20, 2)
    data['MACD'], data['MACD_Signal'] = compute_macd(data, 12, 26, 9)

    # Create lag features
    for lag in range(1, 6):
        data[f'close_lag_{lag}'] = data['close'].shift(lag)
        data[f'volume_lag_{lag}'] = data['volume'].shift(lag)

    data.dropna(inplace=True)
    print("[DEBUG] After dropna, shape:", data.shape)

    # For normalization, drop the target column; here we normalize all features except 'close'
    features_to_normalize = data.drop(columns=['close'])
    scaler = StandardScaler()
    normalized_arr = scaler.fit_transform(features_to_normalize)
    normalized_data = pd.DataFrame(normalized_arr, columns=features_to_normalize.columns)
    # Add back the target column unscaled
    normalized_data['close'] = data['close'].values
    print("[DEBUG] Final shape:", normalized_data.shape)
    return normalized_data, scaler

def run_preprocessing(csv_file_path):
    print("[INFO] Reading CSV:", csv_file_path)
    df = pd.read_csv(csv_file_path)
    print("[INFO] CSV loaded. Shape:", df.shape)
    processed_df, scaler = preprocess_data(df)
    return processed_df, scaler

############################################
#       Model Components for a More Complete TFT
############################################
# 1. Gated Residual Network (GRN)
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gate = self.sigmoid(self.gate(x))
        x = x * gate
        return x + residual

# 2. Variable Selection Network (applied to each time step)
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        self.num_features = input_dim
        # Use a GRN for each feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, output_dim=1) for _ in range(input_dim)
        ])

    def forward(self, x):
        # x shape: (batch, input_dim)
        weights = []
        selected_features = []
        for i in range(self.num_features):
            feature = x[:, i].unsqueeze(1)  # (batch, 1)
            selected = self.feature_grns[i](feature)  # (batch, 1)
            selected_features.append(selected)
            weights.append(selected)
        # Stack weights and normalize them
        weights = torch.cat(weights, dim=1)  # (batch, input_dim)
        norm_weights = torch.softmax(weights, dim=1)
        # Combine original features with normalized weights
        x_selected = (x * norm_weights).sum(dim=1, keepdim=True)  # (batch, 1)
        return x_selected, norm_weights

# 3. Complete Temporal Fusion Transformer with Multi-Head Attention and Gating
class TemporalFusionTransformerComplete(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(TemporalFusionTransformerComplete, self).__init__()
        # We assume input x is shaped (batch, seq_len, input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Variable selection applied on each time step
        self.var_sel = VariableSelectionNetwork(input_dim, hidden_dim)
        # Project the selected feature to hidden dimension
        self.input_projection = nn.Linear(1, hidden_dim)
        # Multi-head attention layer; note: using PyTorch's built-in module.
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        # Gated Residual Network (for post-attention processing)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, output_dim=hidden_dim, dropout=dropout)
        # Final output layer for forecasting next time step's 'close'
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        selected_list = []
        var_sel_weights = []
        # Apply variable selection for each time step independently
        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, input_dim)
            selected, weights = self.var_sel(xt)  # selected: (batch, 1)
            selected_list.append(selected)
            var_sel_weights.append(weights)
        # Stack along time dimension -> (batch, seq_len, 1)
        selected_features = torch.stack(selected_list, dim=1)
        # Project to hidden dimension: (batch, seq_len, hidden_dim)
        proj = self.input_projection(selected_features)
        # Rearrange for multi-head attention: (seq_len, batch, hidden_dim)
        proj = proj.transpose(0, 1)
        attn_output, attn_weights = self.multihead_attn(proj, proj, proj)
        # Restore shape to (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(0, 1)
        # Use the output at the final time step as the representation
        final_repr = attn_output[:, -1, :]  # (batch, hidden_dim)
        final_repr = self.grn(final_repr)
        final_repr = self.dropout(final_repr)
        output = self.output_layer(final_repr)  # (batch, 1)
        return output, attn_weights, var_sel_weights

############################################
#       Training and Evaluation on Daily Data
############################################
def train_model_on_daily(processed_data, scaler):
    # Assume processed_data is the preprocessed daily DataFrame.
    # For the TFT complete, we need to create sequences.
    # Here we create simple sequences by taking a rolling window from the data.
    seq_length = 10  # example sequence length
    features = processed_data.drop(columns=['close'])
    target = processed_data['close'].values
    data_array = features.values  # shape (num_samples, num_features)
    
    X_seq = []
    y_seq = []
    for i in range(len(data_array) - seq_length):
        X_seq.append(data_array[i : i + seq_length])
        y_seq.append(target[i + seq_length])
    X_seq = np.array(X_seq)  # (batch, seq_length, num_features)
    y_seq = np.array(y_seq)  # (batch,)

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Initialize the complete TFT with attention, gating, and variable selection.
    model = TemporalFusionTransformerComplete(input_dim=X_train.shape[2],
                                                hidden_dim=64,
                                                num_heads=4,
                                                dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train for a number of epochs
    epochs = 50
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)  # shape: (batch, seq_length, num_features)
        targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (batch, 1)
        optimizer.zero_grad()
        outputs, attn_weights, var_sel_weights = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate on validation set
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    predictions, _, _ = model(X_val_tensor)
    predictions = predictions.detach().numpy().flatten()
    print("Attention-Enhanced TFT Evaluation on Daily Data:")
    print(f"MSE: {mean_squared_error(y_val, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_val, predictions):.4f}")

    # Save the stacking ensemble or any chosen model for later use
    features_list = list(features.columns)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(features_list, f)
    print("[INFO] Complete TFT model, scaler, and features saved.")

############################################
#       Evaluate Model on Hourly Data
############################################
def evaluate_model(model_path, scaler_path, features_path, hourly_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        print("[INFO] Loading hourly data...")
        hourly_data = pd.read_csv(hourly_path)
        processed_hourly, _ = preprocess_data(hourly_data)
        # Create sequences from hourly data (using same seq_length as training)
        seq_length = 10
        X_seq = []
        y_seq = []
        hourly_features = processed_hourly.drop(columns=['close']).values
        hourly_target = processed_hourly['close'].values
        for i in range(len(hourly_features) - seq_length):
            X_seq.append(hourly_features[i : i + seq_length])
            y_seq.append(hourly_target[i + seq_length])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Make predictions using the loaded model
        model.eval()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        predictions, _, _ = model(X_tensor)
        predictions = predictions.detach().numpy().flatten()
        mse = mean_squared_error(y_seq, predictions)
        mae = mean_absolute_error(y_seq, predictions)
        print(f"[INFO] Hourly Data Evaluation -- MSE: {mse:.4f}, MAE: {mae:.4f}")
    except Exception as e:
        print(f"[ERROR] An error occurred during evaluation: {e}")

############################################
#       Main Execution Block
############################################
if __name__ == "__main__":
    print("[INFO] Starting preprocessing of daily data...")
    processed_daily, scaler = run_preprocessing(DAILY_DATA_PATH)
    if processed_daily.empty:
        print("[ERROR] Processed daily data is empty. Exiting.")
    else:
        print("[INFO] Preprocessing complete. Processed daily data shape:", processed_daily.shape)
        processed_daily.to_csv("bitcoin_daily_processed.csv", index=False)
        # Train the complete TFT with attention, gating, and variable selection on daily data
        train_model_on_daily(processed_daily, scaler)

    # Evaluate the saved model on hourly data
    evaluate_model(MODEL_PATH, SCALER_PATH, FEATURES_PATH, HOURLY_DATA_PATH)
