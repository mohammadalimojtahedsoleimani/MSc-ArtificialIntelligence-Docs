import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist

N = 20
x = np.linspace(0, 2, N).reshape(-1, 1)


def main_function(x):
    return 0.3 * np.cos(3 * np.pi * x) + 0.7 * np.sin(np.pi * x) + 0.5


out_y = main_function(x)
rand_noise = np.random.normal(0, np.sqrt(0.01), N).reshape(-1, 1)
y = out_y + rand_noise


def rbf_model(x, centers, variances):
    distances = cdist(x, centers)
    rbf_values = np.exp(-distances ** 2 / (2 * variances))
    return rbf_values


def fit_rbf_network(x, y, M):
    centers = np.random.choice(x.flatten(), M).reshape(-1, 1)
    variances = np.random.rand(M)

    rbf_features = rbf_model(x, centers, variances)

    model = LinearRegression()
    model.fit(rbf_features, y)
    y_pred = model.predict(rbf_features)

    mse = mean_squared_error(y, y_pred)
    return y_pred, mse


for M in [10, 50]:
    y_pred, mse = fit_rbf_network(x, y, M)
    plt.figure()
    plt.plot(x, out_y, label='Main Function', color='orange')
    plt.scatter(x, y, color='red', label='Noisy Data')
    plt.plot(x, y_pred, label=f'Our RBF network predication with (M={M}) centers', color='green')
    plt.title(f'RBF Network with M={M} centers, mean squared error is={mse:.4f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
