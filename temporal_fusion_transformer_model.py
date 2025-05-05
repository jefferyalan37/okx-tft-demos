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
from temporal_fusion_transformer_model import TemporalFusionTransformerComplete

# Define the model (adjust input_dim and other parameters as needed)
model = TemporalFusionTransformerComplete(input_dim=10, hidden_dim=64, num_heads=4, dropout=0.1)

# Save the model's state
torch.save(model.state_dict(), TFT_PATH)

# Reload the model's state (if needed later)
model.load_state_dict(torch.load(TFT_PATH))
model.eval()

# Define paths
BASE_MODEL_DIR = os.getenv("MODEL_DIR", "./models/")
MODEL_PATH = os.path.join(BASE_MODEL_DIR, "tft_model.pkl")
SCALER_PATH = os.path.join(BASE_MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_MODEL_DIR, "required_features.pkl")
DAILY_DATA_PATH = "./OKXDemoProject-5/data/final_bitcoin_daily_processed.csv"
HOURLY_DATA_PATH = "./OKXDemoProject-5/data/final_bitcoin_hourly_processed.csv"
TFT_PATH = os.path.join(BASE_MODEL_DIR, "tft_state_dict.pt")

# Ensure the model directory exists
os.makedirs(BASE_MODEL_DIR, exist_ok=True)


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
#       Model Components for TFT
############################################
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


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        self.num_features = input_dim
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, output_dim=1) for _ in range(input_dim)
        ])

    def forward(self, x):
        weights = []
        selected_features = []
        for i in range(self.num_features):
            feature = x[:, i].unsqueeze(1)
            selected = self.feature_grns[i](feature)
            selected_features.append(selected)
            weights.append(selected)
        weights = torch.cat(weights, dim=1)
        norm_weights = torch.softmax(weights, dim=1)
        x_selected = (x * norm_weights).sum(dim=1, keepdim=True)
        return x_selected, norm_weights


class TemporalFusionTransformerComplete(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(TemporalFusionTransformerComplete, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.var_sel = VariableSelectionNetwork(input_dim, hidden_dim)
        self.input_projection = nn.Linear(1, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        selected_list = []
        var_sel_weights = []
        for t in range(seq_len):
            xt = x[:, t, :]
            selected, weights = self.var_sel(xt)
            selected_list.append(selected)
            var_sel_weights.append(weights)
        selected_features = torch.stack(selected_list, dim=1)
        proj = self.input_projection(selected_features)
        proj = proj.transpose(0, 1)
        attn_output, attn_weights = self.multihead_attn(proj, proj, proj)
        attn_output = attn_output.transpose(0, 1)
        final_repr = attn_output[:, -1, :]
        final_repr = self.grn(final_repr)
        final_repr = self.dropout(final_repr)
        output = self.output_layer(final_repr)
        return output, attn_weights, var_sel_weights


############################################
#       Main Script
############################################
if __name__ == "__main__":
    print("[INFO] Starting preprocessing of daily data...")
    processed_daily, scaler = run_preprocessing(DAILY_DATA_PATH)
    if processed_daily.empty:
        print("[ERROR] Processed daily data is empty. Exiting.")
    else:
        print("[INFO] Preprocessing complete. Processed daily data shape:", processed_daily.shape)
