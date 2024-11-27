import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Load the Iris dataset
iris_df = pd.read_csv('Iris.csv')

# Inspect the data
print(iris_df.head())

# Preprocessing - Removing 'Id' and 'Species' columns for clustering (we are only interested in features)
X = iris_df.drop(columns=['Id', 'Species'])

# Standardize the data to bring all features to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Create Dendrogram to decide the number of clusters
linked = linkage(X_scaled, method='average')  # Use average linkage

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering (Average Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Step 2: Applying Agglomerative Clustering with 'average' linkage
agglomerative_clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
clusters = agglomerative_clustering.fit_predict(X_scaled)

# Add cluster results to the original dataframe
iris_df['Cluster'] = clusters

# Print the cluster assignments
print(iris_df[['Species', 'Cluster']].head(10))

# Step 3: Evaluating the clustering by comparing with the original labels (for educational purposes)
# Convert species to numerical labels for comparison
iris_df['Species_Label'] = iris_df['Species'].factorize()[0]

# Confusion matrix and accuracy
conf_matrix = confusion_matrix(iris_df['Species_Label'], iris_df['Cluster'])
accuracy = accuracy_score(iris_df['Species_Label'], iris_df['Cluster'])

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nAccuracy: {accuracy*100:.2f}%")

# Step 4: Visualizing the clusters - Scatter plot of Sepal features with custom colors
plt.figure(figsize=(10, 7))
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Cluster', palette=['red', 'yellow', 'blue'], data=iris_df)
plt.title('Cluster Visualization (Sepal Features)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Cluster')
plt.show()

# Step 5: Visualizing the clusters - Scatter plot of Petal features with custom colors
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Cluster', palette=['red', 'yellow', 'blue'], data=iris_df)
plt.title('Cluster Visualization (Petal Features)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Cluster')
plt.show()

# Optional: Test with other linkages (uncomment to run)
# agglomerative_clustering = AgglomerativeClustering(n_clusters=3, linkage='complete')
# clusters = agglomerative_clustering.fit_predict(X_scaled)
