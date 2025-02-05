import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from AdaBoost import AdaBoost, DecisionStump
from Random_Forest import RandomForest, DecisionTree


dataset = openml.datasets.get_dataset(37)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

y_ada = np.where(y.to_numpy() == "tested_negative", -1, 1)
data_ada = pd.concat([X, pd.Series(y_ada, name='class')], axis=1)
X_ada = data_ada.drop('class', axis=1).values
y_ada = data_ada['class'].values

data_rf = pd.concat([X, y], axis=1)
X_rf = data_rf.drop('class', axis=1).values
y_rf = data_rf['class'].values


scaler = StandardScaler()
X_scaled_ada = scaler.fit_transform(X_ada)
X_scaled_rf = scaler.fit_transform(X_rf)


X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(
    X_scaled_ada, y_ada, test_size=0.2, random_state=42
)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_scaled_rf, y_rf, test_size=0.2, random_state=42
)


adaboost_model = AdaBoost(n_clf=20)
adaboost_model.fit(X_train_ada, y_train_ada)
y_pred_ada = adaboost_model.predict(X_test_ada)
accuracy_ada = accuracy_score(y_test_ada, y_pred_ada)
print(f"Accuracy of AdaBoost: {accuracy_ada * 100:.2f}%")


rf_model = RandomForest(n_trees=10, max_depth=5, n_features=2)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print(f"Accuracy of Random Forest: {accuracy_rf * 100:.2f}%")


models = ['AdaBoost', 'Random Forest']
accuracies = [accuracy_ada, accuracy_rf]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: AdaBoost vs. Random Forest')
plt.ylim([0, 1])  # Set y-axis limit to 0-1


for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center')

plt.show()