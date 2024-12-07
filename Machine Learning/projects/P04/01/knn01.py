import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


iris = datasets.load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


k = 5
knn = KNN(k=k)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"KNN Accuracy (k={k}): {accuracy}")

def classify_new_sample():
    try:
        sepal_length = float(sepal_length_entry.get())
        sepal_width = float(sepal_width_entry.get())
        petal_length = float(petal_length_entry.get())
        petal_width = float(petal_width_entry.get())

        new_sample = np.array([sepal_length, sepal_width, petal_length, petal_width])
        prediction = knn.predict(new_sample.reshape(1, -1))[0]
        predicted_class = iris.target_names[prediction]

        result_label.config(text=f"Predicted Class: {predicted_class}")
    except ValueError:
        result_label.config(text="Invalid Input!")

window = tk.Tk()
window.title("Iris Classifier (KNN)")


sepal_length_label = ttk.Label(window, text="Sepal Length (cm):")
sepal_length_label.grid(row=0, column=0, padx=5, pady=5)
sepal_length_entry = ttk.Entry(window)
sepal_length_entry.grid(row=0, column=1, padx=5, pady=5)

sepal_width_label = ttk.Label(window, text="Sepal Width (cm):")
sepal_width_label.grid(row=1, column=0, padx=5, pady=5)
sepal_width_entry = ttk.Entry(window)
sepal_width_entry.grid(row=1, column=1, padx=5, pady=5)

petal_length_label = ttk.Label(window, text="Petal Length (cm):")
petal_length_label.grid(row=2, column=0, padx=5, pady=5)
petal_length_entry = ttk.Entry(window)
petal_length_entry.grid(row=2, column=1, padx=5, pady=5)

petal_width_label = ttk.Label(window, text="Petal Width (cm):")
petal_width_label.grid(row=3, column=0, padx=5, pady=5)
petal_width_entry = ttk.Entry(window)
petal_width_entry.grid(row=3, column=1, padx=5, pady=5)


classify_button = ttk.Button(window, text="Classify", command=classify_new_sample)
classify_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)


result_label = ttk.Label(window, text="")
result_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

window.mainloop()