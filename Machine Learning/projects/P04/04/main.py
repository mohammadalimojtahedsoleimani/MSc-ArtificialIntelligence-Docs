import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



class Node:
    def __init__(self, depth=0, max_depth=5, min_samples_split=10):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.linear_model = None

    def is_leaf(self):
        return self.linear_model is not None



def calculate_rss(y):
    if len(y) == 0:
        return 0
    mean = np.mean(y)
    return np.sum((y - mean) ** 2)


def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_rss = float('inf')
    current_rss = calculate_rss(y)

    n_samples, n_features = X.shape

    for feature in X.columns:
        values = X[feature].unique()
        values.sort()
        for threshold in values:
            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold
            y_left = y[left_mask]
            y_right = y[right_mask]
            rss = calculate_rss(y_left) + calculate_rss(y_right)
            if rss < best_rss:
                best_rss = rss
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_rss



def fit_linear_regression(X, y):

    X_b = np.hstack([np.ones((X.shape[0], 1)), X])

    try:
        beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    except np.linalg.LinAlgError:
        
        beta = np.linalg.pinv(X_b).dot(y)
    return beta


def predict_linear_regression(beta, X):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b.dot(beta)



class LoLiMoT:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.root = Node(depth=0, max_depth=max_depth, min_samples_split=min_samples_split)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self._build_tree(self.root, X, y)

    def _build_tree(self, node, X, y):
        if (node.depth >= self.max_depth) or (len(y) < node.min_samples_split):
            
            node.linear_model = fit_linear_regression(X.values, y.values)
            return

        feature, threshold, rss = best_split(X, y)
        if feature is None:
            node.linear_model = fit_linear_regression(X.values, y.values)
            return

        node.feature_index = feature
        node.threshold = threshold

      
        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        
        if len(y_left) == 0 or len(y_right) == 0:
            node.linear_model = fit_linear_regression(X.values, y.values)
            return

       
        node.left = Node(depth=node.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        node.right = Node(depth=node.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

      
        self._build_tree(node.left, X_left, y_left)
        self._build_tree(node.right, X_right, y_right)

    def predict(self, X):
        predictions = np.array([self._predict_sample(x, self.root) for _, x in X.iterrows()])
        return predictions

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return predict_linear_regression(node.linear_model, x.values.reshape(1, -1))[0]
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)



housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LoLiMoT(max_depth=5, min_samples_split=20)


print("Training LoLiMoT...")
model.fit(X_train, y_train)
print("Training completed.")


print("Making predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions completed.")


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")
