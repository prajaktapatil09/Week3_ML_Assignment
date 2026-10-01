import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Iris Dataset:")
print(data.head())

# Select features
X = data

# Create K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train model and predict clusters
clusters = kmeans.fit_predict(X)

# Add cluster information
data["Cluster"] = clusters

print("\nClustered Data:")
print(data.head())

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot clusters using first two features
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=clusters
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering of Iris Dataset")

plt.savefig("kmeans_iris_clusters.png")
plt.show()
