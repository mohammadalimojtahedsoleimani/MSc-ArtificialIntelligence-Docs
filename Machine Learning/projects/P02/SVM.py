# 4033904504
# Mohammad Ali Mojtahed Soleimani

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.optimize import minimize

file_path = './main.xlsx'
excel_data = pd.ExcelFile(file_path)

data = excel_data.parse('Sheet1')

encoded_data = data.copy()
label_encoders = {}
for column in encoded_data.columns:
    if encoded_data[column].dtype == 'object':
        le = LabelEncoder()
        encoded_data[column] = le.fit_transform(encoded_data[column])
        label_encoders[column] = le


X = encoded_data.drop(columns=['Class']).values
y = encoded_data['Class'].values
y = np.where(y == 0, -1, 1)


def svm_objective(params, X, y, C=1.0):
    w = params[:-1]
    b = params[-1]

    hinge_loss = np.maximum(0, 1 - y * (X.dot(w) + b))

    return 0.5 * np.dot(w, w) + C * hinge_loss.sum()


def svm_constraint(params, X, y):
    w = params[:-1]
    b = params[-1]
    return y * (X.dot(w) + b) - 1


initial_params = np.zeros(X.shape[1] + 1)

result_scipy = minimize(
    svm_objective,
    initial_params,
    args=(X, y),
    method='SLSQP',
    constraints={'type': 'ineq', 'fun': svm_constraint, 'args': (X, y)}
)

optimized_weight = result_scipy.x[:-1]
optimized_bias = result_scipy.x[-1]

def svm_predict(X, w, b):
    return np.sign(X.dot(w) + b)


y_pred = svm_predict(X, optimized_weight, optimized_bias)

accuracy = np.mean(y_pred == y)
error_rate = 1 - accuracy

true_positives = np.sum((y_pred == 1) & (y == 1))
false_positives = np.sum((y_pred == 1) & (y == -1))
false_negatives = np.sum((y_pred == -1) & (y == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print("Accuracy is:", accuracy)
print("error_rate is:", error_rate)
print("precision is:", precision)
print("recall is:", recall)
if precision == 0 and recall == 0:
    print("no true positives were predicted in this setup")
