# okx-tft-demos

🚀 Temporal Fusion Transformer (TFT) + Ensemble AI Pipeline for Institutional Crypto Forecasting

## Overview

This repository contains a full AI/ML pipeline leveraging a **Temporal Fusion Transformer (TFT)** for short- and long-term forecasting of digital asset prices. It's designed for **institutional-grade performance**, including:

- ⚙️ Feature engineering (technical indicators, lag features)
- 🧠 TFT model with attention, gating, and variable selection
- 📈 XGBoost and ensemble blending
- ⏱ Support for both **daily** and **hourly** crypto data (BTC/USD, ETH/USD, etc.)
- 🔁 Retraining + evaluation scripts ready for deployment

---

## 📁 Project Structure
okx-tft-demos/
├── data/ # Raw and preprocessed .csv and .npz files
├── models/ # Saved scalers, models, and TFT state_dict
├── regenerate_models.py # Entry point to rebuild and retrain the full ensemble
├── transformer_model.py # XGBoost + stacking model
├── temporal_fusion_transformer_model.py # Full TFT with attention & variable selection
└── README.md # You're here.
