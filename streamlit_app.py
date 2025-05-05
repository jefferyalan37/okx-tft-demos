import streamlit as st
import pandas as pd
import numpy as np
from temporal_fusion_transformer_model import (
    evaluate_model,
    preprocess_data  # Assuming preprocess_data can be reused
)

# Page Configurations
st.set_page_config(
    page_title="OKX TFT Demo - Hourly Dataset",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
features = [
    "🏠 Overview",
    "📊 View Hourly Dataset",
    "📈 Hourly Predictions"
]
choice = st.sidebar.radio("Select Feature", features)

# Load the Hourly Dataset
@st.cache_data
def load_hourly_data():
    hourly_data_path = "./data/final_bitcoin_hourly_processed.csv"
    return pd.read_csv(hourly_data_path)

# Main Sections
if choice == "🏠 Overview":
    st.title("OKX Temporal Fusion Transformer (TFT) Demo")
    st.markdown("""
    Welcome to the OKX TFT demo! This section focuses on the hourly dataset:
    - View the processed hourly data.
    - Generate model predictions based on the hourly dataset.
    Use the navigation menu to explore.
    """)

elif choice == "📊 View Hourly Dataset":
    st.title("View Hourly Dataset")
    st.markdown("This section displays the processed hourly dataset.")
    try:
        # Load and display dataset
        hourly_data = load_hourly_data()
        st.write("Hourly Dataset:")
        st.dataframe(hourly_data)

        # Visualization
        st.markdown("### Close Price Trend")
        st.line_chart(hourly_data["close"])

        st.markdown("### Volume Trend")
        st.line_chart(hourly_data["volume"])
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")

elif choice == "📈 Hourly Predictions":
    st.title("Hourly Predictions")
    st.markdown("This section runs predictions on the hourly dataset using the TFT model.")
    try:
        # Load dataset
        hourly_data = load_hourly_data()
        
        # Preprocess data
        st.markdown("### Preprocessing Data...")
        processed_data, scaler = preprocess_data(hourly_data)
        if processed_data.empty:
            st.error("Preprocessed data is empty. Please check the dataset.")
        else:
            st.write("Preprocessed Data:")
            st.dataframe(processed_data.head())

            # Run model predictions
            st.markdown("### Generating Predictions...")
            # Replace mock predictions with actual model predictions
            predictions = evaluate_model(
                model_path="./models/tft_model.pkl",  # Path to the pretrained model
                scaler_path="./models/scaler.pkl",    # Path to the scaler
                features_path="./models/required_features.pkl",  # Path to required features
                data_path="./data/final_bitcoin_hourly_processed.csv"  # Path to the hourly dataset
            )

            # Check if predictions were generated successfully
            if predictions is None or len(predictions) == 0:
                st.error("No predictions were generated. Please check the model and data.")
            else:
                processed_data["Predictions"] = predictions

                # Display predictions
                st.write("Hourly Predictions:")
                st.dataframe(processed_data[["close", "Predictions"]])

                # Visualization
                st.markdown("### Actual vs Predicted Close Prices")
                st.line_chart({
                    "Actual": processed_data["close"],
                    "Predicted": processed_data["Predictions"]
                })
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.sidebar.info("Developed by OKX AI Team")
