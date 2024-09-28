# -*- coding: utf-8 -*-


import pandas as pd

print("LOAD THE DATA")
data = pd.read_csv('//azstaeanzprds01.file.core.windows.net/azfsaeanzprds01/PRODUCTION/Personal/Nicole/PYTHON_SQL/mini project 3/ED_wait_time_2022_2023.csv')

print("EDA")
print(data.head())
data.info()
# Check for missing values
print(data.isnull().sum())
print (data.shape)
print(data.columns)
#EDA
# Summary statistics
print(data.describe())

import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the number of presentations by state or territory
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='StateOrTerritory', order=data['StateOrTerritory'].value_counts().index)
plt.title('Number of Presentations by State or Territory')
plt.xlabel('State or Territory')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plotting the distribution of values by state or territory
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='StateOrTerritory', y='Value')
plt.title('Distribution of Values by State or Territory')
plt.xlabel('State or Territory')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# Plotting the distribution of values by time measure
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='TimeMeasure', y='Value')
plt.title('Distribution of Values by Time Measure')
plt.xlabel('Time Measure')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



# Unsupervised Learning: Clustering and Dimensionality Reduction

# Select relevant columns for clustering (excluding categorical columns)
numeric_data = data[['Value']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=1)  # Fixing n_components to 1 due to the error
pca_result = pca.fit_transform(scaled_data)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_result)

# Add PCA results and cluster labels to the original data
data['PCA1'] = pca_result[:, 0]
data['Cluster'] = clusters

# Plot PCA results with cluster labels
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='PCA1', y='Value', hue='Cluster', palette='viridis')
plt.title('PCA Results with K-Means Clusters')
plt.xlabel('PCA1')
plt.ylabel('Value')
plt.show()

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)

# Add t-SNE results to the original data
data['TSNE1'] = tsne_result[:, 0]
data['TSNE2'] = tsne_result[:, 1]

# Plot t-SNE results with cluster labels
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis')
plt.title('t-SNE Results with K-Means Clusters')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()

print("UNSUPERVISED LEARNING")
# Documenting Clusters and Patterns

# Display cluster centers and their potential business implications
cluster_centers = kmeans.cluster_centers_
print("\nCluster Centers (PCA space):")
print(cluster_centers)

# Analyse clusters and their potential business implications
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster} Summary:")
    print(cluster_data.describe())
    print(f"Potential Business Implications for Cluster {cluster}:")
    print("-" * 50)
    print("1. Identify regions with higher wait times and length of stay.")
    print("2. Allocate resources to regions with higher demand.")
    print("3. Implement best practices from regions with lower wait times.")
    print("4. Monitor and improve performance in underperforming regions.")
    print("-" * 50)

print("Feature Engineering and Unsupervised Learning")
# Select relevant columns for clustering (excluding categorical columns)
numeric_data = data[['Value']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=1)  # Fixing n_components to 1 due to the error
pca_result = pca.fit_transform(scaled_data)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_result)

# Add PCA results and cluster labels to the original data
data['PCA1'] = pca_result[:, 0]
data['Cluster'] = clusters

#Build and Evaluate a Predictive Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the target variable (for demonstration purposes, let's predict 'Cluster')
target = 'Cluster'
features = ['Value', 'PCA1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

#Model Without Unsupervised Learning Features
model_without_unsupervised = RandomForestClassifier(random_state=42)
model_without_unsupervised.fit(X_train[['Value']], y_train)
y_pred_without_unsupervised = model_without_unsupervised.predict(X_test[['Value']])
accuracy_without_unsupervised = accuracy_score(y_test, y_pred_without_unsupervised)
print("\nModel Performance without Unsupervised Learning Features:")
print(f"Accuracy: {accuracy_without_unsupervised}")
print(classification_report(y_test, y_pred_without_unsupervised))

#Model With Unsupervised Learning Features
model_with_unsupervised = RandomForestClassifier(random_state=42)
model_with_unsupervised.fit(X_train, y_train)
y_pred_with_unsupervised = model_with_unsupervised.predict(X_test)
accuracy_with_unsupervised = accuracy_score(y_test, y_pred_with_unsupervised)
print("\nModel Performance with Unsupervised Learning Features (PCA1):")
print(f"Accuracy: {accuracy_with_unsupervised}")
print(classification_report(y_test, y_pred_with_unsupervised))

# Compare model performance with and without unsupervised learning features
print("\nComparison of Model Performance:")
print(f"Accuracy without Unsupervised Learning Features: {accuracy_without_unsupervised}")
print(f"Accuracy with Unsupervised Learning Features (PCA1): {accuracy_with_unsupervised}")








