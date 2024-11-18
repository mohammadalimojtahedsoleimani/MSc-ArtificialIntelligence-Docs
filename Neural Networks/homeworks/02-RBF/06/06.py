import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


def rbf(x, center, sigma):
    if sigma <= 0:
        sigma = abs(sigma)
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def target_function(x):
    return np.sin(x) + np.cos(2 * x) - 3 * x + 1


x_train = np.linspace(-5, 5, 250)
y_train = target_function(x_train)
centers = np.linspace(-5, 5, 5)


def create_rbf_features(x, centers, widths):
    features = np.zeros((len(x), len(centers)))
    for i, center in enumerate(centers):
        features[:, i] = rbf(x, center, widths[i])
    return features


def loss_function(params):
    widths = params
    X_rbf = create_rbf_features(x_train, centers, widths)
    weights = np.linalg.pinv(X_rbf) @ y_train
    y_pred = X_rbf @ weights
    return mean_squared_error(y_train, y_pred)


initial_widths = np.ones(len(centers)) * 2.0

result = minimize(loss_function, initial_widths, method='BFGS')

optimized_widths = result.x
print("Optimized Widths:", optimized_widths)

X_rbf = create_rbf_features(x_train, centers, optimized_widths)

weights = np.linalg.pinv(X_rbf) @ y_train


def predict(x, centers, widths, weights):
    features = create_rbf_features(x, centers, widths)
    return features @ weights


x_test = np.linspace(-5, 5, 1000)
y_pred = predict(x_test, centers, optimized_widths, weights)
y_true = target_function(x_test)

plt.figure(figsize=(12, 6))
plt.plot(x_test, y_true, 'b-', label='Original Function')
plt.plot(x_test, y_pred, 'r--', label='RBF Approximation')
plt.plot(centers, np.zeros_like(centers), 'go', label='RBF Centers')
plt.grid(True)
plt.legend()
plt.title('RBF Network Approximation vs Original Function')
plt.xlabel('x')
plt.ylabel('y')

mse = mean_squared_error(y_true, predict(x_test, centers, optimized_widths, weights))
print(f"Mean Squared Error: {mse:.6f}")

plt.figure(figsize=(12, 6))
for i, center in enumerate(centers):
    rbf_contribution = weights[i] * rbf(x_test, center, optimized_widths[i])
    plt.plot(x_test, rbf_contribution, '--', label=f'RBF {i + 1}')
plt.grid(True)
plt.legend()
plt.title('Individual RBF Contributions')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
