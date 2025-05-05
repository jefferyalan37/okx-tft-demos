from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Add, Layer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Define the custom GetItem layer
class GetItem(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Define the model
def build_and_save_model():
    input_layer = Input(shape=(30, 1), name="input_layer")
    dense = Dense(64, activation="relu")(input_layer)
    attention = MultiHeadAttention(num_heads=2, key_dim=32)(dense, dense)
    add = Add()([dense, attention])
    norm = LayerNormalization()(add)
    dense_out = Dense(128, activation="relu")(norm)
    dense_out2 = Dense(64, activation="relu")(dense_out)
    add_2 = Add()([norm, dense_out2])
    norm_2 = LayerNormalization()(add_2)
    get_item = GetItem(index=0)(norm_2)
    output_layer = Dense(1)(get_item)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Save the model in the new format
    model.save("/Users/jeffery/transformer_model.keras", include_optimizer=False)
    print("Model saved to /Users/jeffery/transformer_model.keras")


# Test the model
def test_model():
    # Register the custom layer
    custom_objects = {"GetItem": GetItem}

    # Load the saved model
    model = load_model('/Users/jeffery/transformer_model.keras', custom_objects=custom_objects)

    # Load test data
    data = np.load('/Users/jeffery/prepared_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    # Make predictions
    predictions = model.predict(X_test)

    # Plot predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.show()


# Main script
if __name__ == "__main__":
    # Build and save the model
    build_and_save_model()

    # Test the model
    test_model()
