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

# Select features
X = data

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Add cluster information
data["Cluster"] = clusters

print("Iris Flower Clustering:")
print(data.head())

# Compare actual species with predicted clusters
comparison = pd.DataFrame({
    "Actual Species": iris.target,
    "Predicted Cluster": clusters
})

print("\nActual Species vs Predicted Clusters:")
print(comparison.head(20))

# Plot clusters
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=clusters
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Clustering using K-Means")

plt.savefig("iris_flower_clusters.png")
plt.show()
