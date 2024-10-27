import numpy as np
from typing import List, Tuple

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Initialize weights with small random values
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) - 0.6
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        current_activation = X
        activations = [X]
        layer_inputs = []

        for w, b in zip(self.weights, self.biases):
            layer_input = np.dot(current_activation, w) + b
            layer_inputs.append(layer_input)
            current_activation = self.sigmoid(layer_input)
            activations.append(current_activation)

        return activations, layer_inputs

    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = y.shape[0]
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Calculate initial error
        delta = activations[-1] - y
        
        # Iterate through layers backwards
        for l in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            if l > 0:
                # Calculate error for next iteration
                delta = np.dot(delta, self.weights[l].T) * self.sigmoid_derivative(activations[l])

        return weight_gradients, bias_gradients

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> List[float]:
        errors = []
        
        for _ in range(epochs):
            # Forward propagation
            activations, _ = self.forward(X)
            
            # Calculate error
            error = np.mean(np.square(activations[-1] - y))
            errors.append(error)
            
            # Backward propagation
            weight_gradients, bias_gradients = self.backward(X, y, activations)
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]
                
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:

        activations, _ = self.forward(X)
        return activations[-1]

# Example usage:
# Create a neural network with 2 inputs, 3 hidden nodes, and 1 output
nn = NeuralNetwork([2, 8, 1], learning_rate=0.1)

# Example training data
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # Input
y = np.array([[0], [1], [0], [0]])              # Target (XOR function)

# Train the network
errors = nn.train(X, y, epochs=10000)

# Make predictions
predictions = nn.predict(X)

for i in range(len(X)):
    input_data = X[i]
    target = y[i][0]
    predicted = predictions[i][0]
    print(f"Input: {input_data}, Target: {target:.0f}, Predicted: {predicted:.4f}")

# Calculate accuracy
correct = np.sum(np.round(predictions) == y)
accuracy = correct / len(y) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
