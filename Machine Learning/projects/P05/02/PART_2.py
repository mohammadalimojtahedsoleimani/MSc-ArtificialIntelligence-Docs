import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score



def euclidean_distance(point1, point2):
    return np.sum(point1 != point2)

def initialize_centroids(X, k):

    random_indices = np.random.choice(X.shape[0], k, replace=False)
    return X[random_indices]

def assign_to_clusters(X, centroids):

    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points_in_cluster = X[clusters == i]
        if len(points_in_cluster) > 0:
            for j in range(X.shape[1]):
                values, counts = np.unique(points_in_cluster[:, j], return_counts=True)
                mode_index = np.argmax(counts)
                new_centroids[i, j] = values[mode_index]
        else:
            
            new_centroids[i] = X[np.random.choice(X.shape[0])]
    return new_centroids

def kmeans(X, k, max_iters=100, num_initializations=10):

    best_clusters = None
    best_centroids = None
    best_metrics = {
        "accuracy": -1,
        "precision": -1,
        "recall": -1
    }

    for _ in range(num_initializations):
        centroids = initialize_centroids(X, k)

        for _ in range(max_iters):
            clusters = assign_to_clusters(X, centroids)
            new_centroids = update_centroids(X, clusters, k)

            if np.array_equal(centroids, new_centroids):
                break

            centroids = new_centroids

        
        metrics = evaluate_clustering(y, clusters, centroids)

        
        if metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            best_clusters = clusters
            best_centroids = centroids

    return best_clusters, best_centroids, best_metrics



def load_and_preprocess(filepath):
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, None

    
    X = df.drop('Class', axis=1)  
    y = df['Class']


    label_encoder = LabelEncoder()
    for col in X.columns:
        X[col] = label_encoder.fit_transform(X[col])

    
    y = label_encoder.fit_transform(y)

    return X.values, y  


def evaluate_clustering(y, clusters, centroids):
    n_clusters = len(centroids)
    cluster_labels = np.zeros(n_clusters)

    
    for i in range(n_clusters):
        true_labels_in_cluster = y[clusters == i]
        if len(true_labels_in_cluster) > 0:
            most_common_label = np.bincount(true_labels_in_cluster).argmax()
            cluster_labels[i] = most_common_label
        else:
            
            cluster_labels[i] = 0  

   
    predicted_labels = np.array([cluster_labels[cluster_id] for cluster_id in clusters])

  
    accuracy = accuracy_score(y, predicted_labels)
    precision = precision_score(y, predicted_labels, zero_division=0)  
    recall = recall_score(y, predicted_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }



if __name__ == "__main__":
    filepath = 'breast_cancer_data.xlsx'  
    X, y = load_and_preprocess(filepath)

    if X is not None and y is not None:
        k = 2 
        clusters, centroids, metrics = kmeans(X, k, num_initializations=100)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Loss Rate (Error Rate): {1 - metrics['accuracy']:.4f}")