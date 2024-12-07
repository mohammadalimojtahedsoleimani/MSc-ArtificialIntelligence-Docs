import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X, y = housing.data, housing.target

scaler = StandardScaler()
X = scaler.fit_transform(X)


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)


class RBFNetwork:
    def __init__(self, num_centers, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf_kernel(self, x, center):
        return np.exp(-np.sum((x - center) ** 2) / (2 * self.sigma ** 2))

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, center in enumerate(self.centers):
                G[i, j] = self._rbf_kernel(x, center)
        return G

    def fit(self, X, y):
        random_indices = np.random.choice(X.shape[0], self.num_centers, replace=False)
        self.centers = X[random_indices]

        G = self._calculate_interpolation_matrix(X)

        self.weights = np.linalg.pinv(G) @ y

    def predict(self, X):

        G = self._calculate_interpolation_matrix(X)
        predictions = G @ self.weights
        return predictions


num_centers = 200
sigma = 2.0

rbf_net = RBFNetwork(num_centers, sigma)
rbf_net.fit(X_train, y_train)

y_pred_train = rbf_net.predict(X_train)
y_pred_test = rbf_net.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Training MSE: {mse_train:.4f}")
print(f"Testing MSE: {mse_test:.4f}")
