###############################################################
# Project : StyleMatch AI
# File    : model_training.py
# Purpose : Fashion Product Recommendation using
#           Hierarchical Clustering
###############################################################

# ==========================================================
# STEP 1 : Import Libraries
# ==========================================================
import joblib

import os
import random
import shutil

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

import matplotlib.pyplot as plt

# ==========================================================
# STEP 2 : Dataset Configuration
# ==========================================================

# Original dataset folder
DATASET_PATH = "Dataset/images"

# Folder that will contain sampled images
SAMPLE_PATH = "Dataset/sample_images"

# Number of images to use
NUM_IMAGES = 1000

# Random seed for reproducibility
RANDOM_STATE = 42

# ==========================================================
# STEP 3 : Create Sample Folder
# ==========================================================

# Create folder if it doesn't exist
os.makedirs(SAMPLE_PATH, exist_ok=True)

# ==========================================================
# STEP 4 : Read Original Dataset
# ==========================================================

print("=" * 60)
print("STEP 1 : Reading Dataset")
print("=" * 60)

image_files = [
    file
    for file in os.listdir(DATASET_PATH)
    if file.lower().endswith(".jpg")
]

print(f"\nTotal Images Found : {len(image_files)}")

# Check whether enough images are available
if len(image_files) < NUM_IMAGES:
    raise ValueError(
        f"\nDataset contains only {len(image_files)} images.\n"
        f"Requested {NUM_IMAGES} images."
    )

# ==========================================================
# STEP 5 : Check Existing Sample Folder
# ==========================================================

existing_images = [
    file
    for file in os.listdir(SAMPLE_PATH)
    if file.lower().endswith(".jpg")
]

# ==========================================================
# STEP 6 : Recreate Subset if Needed
# ==========================================================

if len(existing_images) != NUM_IMAGES:

    print("\nCreating new image subset...")

    # Delete old images
    for image in existing_images:

        os.remove(os.path.join(SAMPLE_PATH, image))

    # Generate same subset every time
    random.seed(RANDOM_STATE)

    sampled_images = random.sample(image_files, NUM_IMAGES)

    # Copy images
    for image in sampled_images:

        source = os.path.join(DATASET_PATH, image)
        destination = os.path.join(SAMPLE_PATH, image)

        shutil.copy(source, destination)

    print("Subset created successfully!")

else:

    print("\nSubset already exists.")

# ==========================================================
# STEP 7 : Final Verification
# ==========================================================

sample_images = [
    file
    for file in os.listdir(SAMPLE_PATH)
    if file.lower().endswith(".jpg")
]

# ==========================================================
# STEP 8 : Dataset Summary
# ==========================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Original Dataset Images : {len(image_files)}")
print(f"Training Dataset Images : {len(sample_images)}")
print(f"Dataset Folder          : {SAMPLE_PATH}")

print("=" * 60)

print("\nDataset is ready for training.\n")

# ==========================================================
# STEP 9 : Image Preprocessing
# ==========================================================

print("=" * 60)
print("STEP 2 : Image Preprocessing")
print("=" * 60)

# Store processed images
processed_images = []

# Store image paths
image_paths = []

# Read every sampled image
sample_images = sorted(os.listdir(SAMPLE_PATH))

for image_name in sample_images:

    image_path = f"{SAMPLE_PATH}/{image_name}"

    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Skip corrupted images
    if image is None:
        continue

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image for ResNet50
    image = cv2.resize(image, (224, 224))

    # Convert image into NumPy array
    image = img_to_array(image)

    # Apply ResNet50 preprocessing
    image = preprocess_input(image)

    processed_images.append(image)
    image_paths.append(image_path)


# Convert list to NumPy array
processed_images = np.array(processed_images)

print("\nImage preprocessing completed successfully.")

print(f"\nTotal Processed Images : {len(processed_images)}")

print(f"Image Tensor Shape     : {processed_images.shape}")

print("=" * 60)

# ==========================================================
# STEP 10 : Load ResNet50 Model
# ==========================================================

print("=" * 60)
print("STEP 3 : Loading ResNet50 Model")
print("=" * 60)

# Load pretrained ResNet50 without the classification layer
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("\nResNet50 Loaded Successfully.")

print("=" * 60)

# ==========================================================
# STEP 11 : Feature Extraction
# ==========================================================

print("Extracting Deep Features...\n")

# Generate feature vectors
features = base_model.predict(
    processed_images,
    batch_size=32,
    verbose=1
)

print("\nFeature Extraction Completed Successfully!")

print(f"\nFeature Matrix Shape : {features.shape}")

print("=" * 60)

# ==========================================================
# STEP 11.1 : L2 Normalize Features
# ==========================================================
# Raw ResNet50 features are dominated by overall brightness /
# color intensity / texture magnitude, NOT garment category or
# shape. This causes unrelated categories (e.g. shoes vs shirts)
# to appear "similar" just because they're visually bright/plain.
#
# L2-normalizing puts every feature vector on the same scale
# (unit length), so cosine similarity reflects the SHAPE of the
# feature vector (structural/semantic content) instead of being
# dominated by magnitude. This must be done consistently here
# AND in app.py's extract_features(), or the two spaces won't
# match.
# ==========================================================

print("=" * 60)
print("STEP 4a : L2 Normalizing Features")
print("=" * 60)

features = normalize(features, norm="l2")

print("\nFeature Normalization Completed Successfully!")
print("=" * 60)

# ==========================================================
# STEP 12 : Feature Scaling
# ==========================================================

print("=" * 60)
print("STEP 4 : Feature Scaling")
print("=" * 60)

# Create StandardScaler object
scaler = StandardScaler()

# Scale the extracted (already normalized) features
scaled_features = scaler.fit_transform(features)

print("\nFeature Scaling Completed Successfully!")

print(f"\nScaled Feature Shape : {scaled_features.shape}")

print("=" * 60)

# ==========================================================
# STEP 13 : PCA for Clustering
# ==========================================================

print("=" * 60)
print("STEP 5 : Applying PCA")
print("=" * 60)

# Keep 100 principal components
pca = PCA(
    n_components=100,
    random_state=42
)

reduced_features = pca.fit_transform(scaled_features)

print("\nPCA Completed Successfully!")

print(f"\nOriginal Shape : {scaled_features.shape}")

print(f"Reduced Shape  : {reduced_features.shape}")

print("=" * 60)

# ==========================================================
# STEP 14 : Generate Dendrogram
# ==========================================================

print("=" * 60)
print("STEP 6 : Generating Dendrogram")
print("=" * 60)

# Generate linkage matrix
linked = linkage(
    reduced_features,
    method="ward"
)

plt.figure(figsize=(15,8))

dendrogram(
    linked,
    truncate_mode="level",
    p=5
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")

plt.grid(True)

plt.show()

# ==========================================================
# STEP 15 : Train Hierarchical Clustering
# ==========================================================

print("=" * 60)
print("STEP 7 : Training Hierarchical Clustering")
print("=" * 60)

model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=150,
    metric="euclidean",
    linkage="ward"
)

cluster_labels = model.fit_predict(
    reduced_features
)


NUM_CLUSTERS = len(
    np.unique(cluster_labels)
)


print(
    f"\nClusters Automatically Identified : {NUM_CLUSTERS}"
)

print("\nHierarchical Clustering Completed!")

print(f"\nNumber of Clusters : {NUM_CLUSTERS}")

print("=" * 60)

# ==========================================================
# STEP 16 : Evaluation
# ==========================================================

print("=" * 60)
print("STEP 8 : Evaluating Model")
print("=" * 60)

score = silhouette_score(
    reduced_features,
    cluster_labels
)

print(f"\nSilhouette Score : {score:.4f}")

print("=" * 60)

# ==========================================================
# STEP 17 : PCA for Visualization
# ==========================================================

pca_visual = PCA(
    n_components=2,
    random_state=42
)

visual_features = pca_visual.fit_transform(reduced_features)

# ==========================================================
# STEP 18 : Cluster Visualization
# ==========================================================

print("=" * 60)
print("STEP 9 : Visualizing Clusters")
print("=" * 60)

plt.figure(figsize=(12,8))

scatter = plt.scatter(
    visual_features[:,0],
    visual_features[:,1],
    c=cluster_labels,
    cmap="tab10",
    s=35
)

plt.title(
    "Fashion Product Clusters using Hierarchical Clustering",
    fontsize=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.legend(
    *scatter.legend_elements(),
    title="Clusters"
)

plt.show()

# ==========================================================
# STEP 19.1 : Calculate Cluster Centers
# ==========================================================

cluster_centers = []

for cluster in range(NUM_CLUSTERS):

    center = reduced_features[
        cluster_labels == cluster
    ].mean(axis=0)

    cluster_centers.append(center)


cluster_centers = np.array(
    cluster_centers
)


np.save(
    "cluster_centers.npy",
    cluster_centers
)

# ==========================================================
# STEP 19 : Save Required Files
# ==========================================================

print("=" * 60)
print("STEP 10 : Saving Models")
print("=" * 60)

# Save Standard Scaler
joblib.dump(scaler, "scaler.pkl")

# Save PCA Model
joblib.dump(pca, "pca.pkl")

# Save original ResNet50 features
np.save(
    "image_features.npy",
    features
)

# Save PCA reduced features for analysis
np.save(
    "pca_features.npy",
    reduced_features
)

# Save Image Paths
np.save("image_paths.npy", image_paths)

# Save cluster assignments
np.save(
    "cluster_labels.npy",
    cluster_labels
)

print("\nFiles Saved Successfully!")

print("\nSaved Files")

print("scaler.pkl")

print("pca.pkl")

print("features.npy")

print("image_paths.npy")

print("=" * 60)
