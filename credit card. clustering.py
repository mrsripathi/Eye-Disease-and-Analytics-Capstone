import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\sripa\Desktop\Guvi Final Project\Data (2)\Data\Credit Card_Clustering\Credit Card_Clustering.csv")

# Step 1: Drop non-numeric columns (like CUST_ID)
df = df.drop("CUST_ID", axis=1)

# Step 2: Choose the imputer method (Try all and compare)
imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "knn": KNNImputer(n_neighbors=5),
    "iterative": IterativeImputer(random_state=0)
}

# To store silhouette scores for different imputers
imputer_scores = {}

# Loop through different imputers and perform clustering
for name, imputer in imputers.items():
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('cluster', KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    
    # Apply the imputer and fit the pipeline
    X_imputed = imputer.fit_transform(df)
    
    # Standardize the data
    X_scaled = pipeline.named_steps['scaler'].transform(X_imputed)
    
    # Perform clustering
    cluster_labels = pipeline.named_steps['cluster'].fit_predict(X_scaled)
    
    # Calculate silhouette score
    score = silhouette_score(X_scaled, cluster_labels)
    imputer_scores[name] = score

    print(f"Silhouette score for {name} imputer: {score}")


# Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()



# Using KNNImputer (as an example, you can switch to any other imputer)
imputer = KNNImputer(n_neighbors=5)
model = KMeans(n_clusters=3, random_state=42)

# Impute missing values and scale the data
X_imputed = imputer.fit_transform(df)
X_scaled = StandardScaler().fit_transform(X_imputed)

# Fit the KMeans model
model.fit(X_scaled)

# Perform PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Get cluster labels
cluster_labels = model.labels_
df['Cluster']= model.labels_

Cluster_profile = df.groupby('Cluster').mean()


# Plot the PCA result with cluster labels
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('KMeans Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()  # Color bar to indicate clusters
plt.show()

