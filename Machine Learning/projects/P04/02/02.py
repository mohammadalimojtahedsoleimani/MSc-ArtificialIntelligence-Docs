import numpy as np

class1 = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [7, 3], [8, 4]])
class2 = np.array([[2, 6], [3, 7], [6, 1], [7, 2], [8, 3], [9, 4]])

data = np.concatenate((class1, class2))
labels = np.array([0] * len(class1) + [1] * len(class2))


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn_classifier(data, labels, test_point, k):
    distances = []
    for i, point in enumerate(data):
        dist = euclidean_distance(test_point, point)
        distances.append((dist, labels[i]))

    distances.sort()
    neighbors = distances[:k]

    class_votes = {}
    for _, label in neighbors:
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1

    return max(class_votes, key=class_votes.get)


def calculate_error(data, labels, k):
    errors = 0
    for i in range(len(data)):
        test_point = data[i]
        test_label = labels[i]
        training_data = np.delete(data, i, axis=0)
        training_labels = np.delete(labels, i)

        predicted_label = knn_classifier(training_data, training_labels, test_point, k)
        if predicted_label != test_label:
            errors += 1
    return errors / len(data)


max_k = len(data) - 1
for k in range(1, max_k + 1):
    error_rate = calculate_error(data, labels, k)
    print(f"k = {k}: Error Rate = {error_rate:.4f}")
