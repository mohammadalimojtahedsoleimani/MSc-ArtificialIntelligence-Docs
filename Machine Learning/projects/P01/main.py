# Mohammad Ali Mojtahed Soleimani
# 4033904504


import numpy as np
import pandas as pd
from typing import List, Callable, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ActivationFunctions:

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sx = ActivationFunctions.sigmoid(x)
        return sx * (1 - sx)

    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


class Layer:
    def __init__(self, input_size: int, output_size: int,
                 activation_function: str = 'sigmoid'):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.activation_name = activation_function

        if activation_function == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation_function == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative

        self.input = None
        self.output = None
        self.input_before_activation = None


class MultiLayerPerceptron:
    def __init__(self, layer_sizes: List[int],
                 activation_functions: List[str],
                 learning_rate: float = 0.01):

        self.layers = []
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i + 1],
                      activation_functions[i])
            )

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:

        current_input = X

        for layer in self.layers:
            layer.input = current_input
            layer.input_before_activation = np.dot(current_input, layer.weights) + layer.bias
            layer.output = layer.activation(layer.input_before_activation)
            current_input = layer.output

        return current_input

    def backward_propagation(self, X: np.ndarray, y: np.ndarray):

        m = X.shape[0]

        delta = self.layers[-1].output - y

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if i > 0:
                prev_layer = self.layers[i - 1]
                input_data = prev_layer.output
            else:
                input_data = X


            delta = delta * layer.activation_derivative(layer.input_before_activation)
            layer.weights_grad = np.dot(input_data.T, delta) / m
            layer.bias_grad = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, layer.weights.T)

            layer.weights -= self.learning_rate * layer.weights_grad
            layer.bias -= self.learning_rate * layer.bias_grad

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int, batch_size: int = 32,
              verbose: bool = True) -> List[float]:

        losses = []
        m = X.shape[0]

        for epoch in range(epochs):

            for i in range(0, m, batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                output = self.forward_propagation(batch_X)

                self.backward_propagation(batch_X, batch_y)

                loss = np.mean(-batch_y * np.log(output + 1e-8) -
                               (1 - batch_y) * np.log(1 - output + 1e-8))
                losses.append(loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:

        probabilities = self.forward_propagation(X)
        return (probabilities >= threshold).astype(int)


def preprocess_data(df: pd.DataFrame) -> tuple:
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    X = df_encoded.drop('HeartDisease', axis=1).values
    y = df_encoded['HeartDisease'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def train_heart_disease_model(df: pd.DataFrame):
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_size = X_train.shape[1]
    layer_sizes = [input_size, 32, 16, 1]
    activation_functions = ['relu', 'relu', 'sigmoid']

    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activation_functions=activation_functions,
        learning_rate=0.2
    )

    losses = mlp.train(X_train, y_train, epochs=1000, batch_size=32)

    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return mlp


df = pd.read_csv('heart.csv')

model = train_heart_disease_model(df)
