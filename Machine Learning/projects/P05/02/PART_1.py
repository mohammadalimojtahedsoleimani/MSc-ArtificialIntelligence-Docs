import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_and_preprocess(filepath):
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None, None, None

    X = df.drop('Class', axis=1)
    y = df['Class']

    label_encoder = LabelEncoder()
    for col in X.columns:
        X[col] = label_encoder.fit_transform(X[col])

    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = None
        self.feature_probs = None
        self.classes = None
        self.y = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.y = y

        self.class_probs = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_probs[i] = np.sum(y == c) / n_samples

        self.feature_probs = {}
        for feature_index in range(n_features):
            self.feature_probs[feature_index] = {}
            for class_index in range(n_classes):
                self.feature_probs[feature_index][class_index] = {}
                X_c = X[y == self.classes[class_index]]
                feature_values = np.unique(X[:, feature_index])
                for value in feature_values:
                    count = np.sum(X_c[:, feature_index] == value)
                    total = X_c.shape[0]
                    self.feature_probs[feature_index][class_index][value] = (
                            (count + self.alpha) / (total + self.alpha * len(feature_values))
                    )

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.class_probs[i])
            likelihood = 0
            for feature_index, feature_value in enumerate(x):
                if feature_value in self.feature_probs[feature_index][i]:
                    likelihood += np.log(self.feature_probs[feature_index][i][feature_value])
                else:
                    likelihood += np.log(
                        self.alpha / (np.sum(self.y == c) + self.alpha * len(self.feature_probs[feature_index][i])))

            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test.values)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    filepath = 'breast_cancer_data.xlsx'
    X_train, X_test, y_train, y_test = load_and_preprocess(filepath)

    if X_train is not None:
        nb_model = NaiveBayes(alpha=1.0)
        nb_model.fit(X_train.values, y_train)

        metrics = evaluate_model(nb_model, X_test, y_test)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Loss Rate (Error Rate): {1 - metrics['accuracy']:.4f}")
