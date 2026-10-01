import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Original Iris Dataset:")
print(data.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply PCA and reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create PCA DataFrame
pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

print("\nPCA Result:")
print(pca_data.head())

# Explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plot PCA result
plt.scatter(
    pca_data["PC1"],
    pca_data["PC2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Iris Dataset")

plt.savefig("pca_iris.png")
plt.show()