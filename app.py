###############################################################
# Project : StyleCluster AI
# File    : app.py
# Purpose : AI-Based Fashion Recommendation System
###############################################################


# ==========================================================
# STEP 1 : Import Libraries
# ==========================================================

import streamlit as st
import numpy as np
import cv2
import os
import joblib

from PIL import Image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize



# ==========================================================
# STEP 2 : Page Configuration
# ==========================================================

st.set_page_config(
    page_title="StyleCluster AI",
    page_icon="👕",
    layout="wide"
)



# ==========================================================
# STEP 3 : Custom CSS
# ==========================================================

st.markdown(
    """
    <style>

    .title {
        text-align:center;
        font-size:45px;
        font-weight:700;
        margin-bottom:5px;
    }

    .subtitle {
        text-align:center;
        font-size:22px;
        color:#555;
        margin-bottom:30px;
    }

    .section {
        text-align:center;
        font-size:28px;
        font-weight:600;
        margin-top:30px;
        margin-bottom:20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ==========================================================
# STEP 4 : Configuration
# ==========================================================

# Minimum cosine similarity required for a product to be
# considered a genuine match. Anything below this is treated
# as "not found" instead of being force-matched.
#
# NOTE: This value depends on your dataset's feature space.
# Turn on DEBUG_MODE below once to see real similarity numbers
# for your catalog, then set this accordingly and turn DEBUG_MODE
# back off.
MIN_SIMILARITY = 0.15

# When True, shows a one-time diagnostic panel in the sidebar
# with real similarity score stats from your own catalog, so you
# can pick MIN_SIMILARITY based on actual numbers instead of a
# guess. Set back to False for normal use.
DEBUG_MODE = False



# ==========================================================
# STEP 5 : Load Saved Files
# ==========================================================

@st.cache_resource
def load_resources():

    # Load ResNet50 raw features for all products
    image_features = np.load("image_features.npy")

    # Load image paths
    image_paths = np.load("image_paths.npy", allow_pickle=True)

    # Load scaler + PCA used during training
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    # Load hierarchical clustering results
    cluster_labels = np.load("cluster_labels.npy")
    cluster_centers = np.load("cluster_centers.npy")

    # Recreate the PCA-reduced feature space for all products
    # (this is the SAME space the clustering was trained on,
    # and the SAME space we must compare the uploaded image in)
    scaled_all = scaler.transform(image_features)
    reduced_all = pca.transform(scaled_all)

    # Load ResNet50 feature extractor
    model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    return (
        image_features,
        image_paths,
        model,
        scaler,
        pca,
        cluster_labels,
        cluster_centers,
        reduced_all
    )



(
    image_features,
    image_paths,
    model,
    scaler,
    pca,
    cluster_labels,
    cluster_centers,
    reduced_all
) = load_resources()



# ==========================================================
# STEP 5.1 : Optional Debug Panel (Similarity Diagnostics)
# ==========================================================
# Shows real similarity score stats from your own catalog so
# you can pick a sensible MIN_SIMILARITY instead of guessing.
# Only runs when DEBUG_MODE = True above.
# ==========================================================

if DEBUG_MODE:

    with st.sidebar:

        st.markdown("### 🔧 Similarity Diagnostics")

        n_samples = min(50, len(reduced_all))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(reduced_all), size=n_samples, replace=False)

        best_match_scores = []

        for i in sample_idx:
            query = reduced_all[i].reshape(1, -1)
            sims = cosine_similarity(query, reduced_all)[0]
            sims[i] = -1  # exclude self-match
            best_match_scores.append(sims.max())

        best_match_scores = np.array(best_match_scores)

        st.write(f"Min : {best_match_scores.min():.4f}")
        st.write(f"Max : {best_match_scores.max():.4f}")
        st.write(f"Mean : {best_match_scores.mean():.4f}")
        st.write(f"Median : {np.median(best_match_scores):.4f}")
        st.write(f"25th pct : {np.percentile(best_match_scores, 25):.4f}")
        st.write(f"75th pct : {np.percentile(best_match_scores, 75):.4f}")

        st.info(
            "Set MIN_SIMILARITY at or below the 25th percentile "
            "value so real catalog matches aren't filtered out."
        )



# ==========================================================
# STEP 6 : Extract Image Features
# ==========================================================

def extract_features(image):

    image = image.convert("RGB")

    image = np.array(image)

    image = cv2.resize(image, (224, 224))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    features = model.predict(image, verbose=0)

    # IMPORTANT: must match the normalization applied during
    # training (model_training.py), or the uploaded image and
    # the catalog won't be in the same comparable space.
    features = normalize(features, norm="l2")

    return features



# ==========================================================
# STEP 7 : Find Similar Products (Cluster-Aware)
# ==========================================================

def recommend_products(uploaded_features, top_k=10, min_similarity=MIN_SIMILARITY):

    # ------------------------------------------------------
    # Bring uploaded image into the SAME feature space used
    # for training the hierarchical clustering model
    # (raw ResNet50 -> StandardScaler -> PCA)
    # ------------------------------------------------------

    scaled_upload = scaler.transform(uploaded_features)
    reduced_upload = pca.transform(scaled_upload)

    # ------------------------------------------------------
    # STEP A : Assign uploaded image to the nearest cluster
    # ------------------------------------------------------

    distances_to_centers = np.linalg.norm(
        cluster_centers - reduced_upload,
        axis=1
    )

    assigned_cluster = np.argmin(distances_to_centers)

    # ------------------------------------------------------
    # STEP B : Get all products belonging to that cluster
    # ------------------------------------------------------

    cluster_member_indices = np.where(
        cluster_labels == assigned_cluster
    )[0]

    # Fallback: if the assigned cluster is too small,
    # widen the search to the entire catalog
    if len(cluster_member_indices) < top_k:
        cluster_member_indices = np.arange(len(image_paths))

    # ------------------------------------------------------
    # STEP C : Rank cluster members by similarity
    # ------------------------------------------------------

    cluster_reduced_features = reduced_all[cluster_member_indices]

    similarity_scores = cosine_similarity(
        reduced_upload,
        cluster_reduced_features
    )[0]

    ranked_order = np.argsort(similarity_scores)[::-1][:top_k]

    ranked_scores = similarity_scores[ranked_order]

    similar_indices = cluster_member_indices[ranked_order]

    # ------------------------------------------------------
    # STEP D : Filter out weak matches
    # ------------------------------------------------------
    # Without this, argsort/argmin will ALWAYS return "closest"
    # items even if nothing in the catalog is actually similar
    # (e.g. uploading an unrelated photo). This threshold makes
    # "no match found" possible.
    # ------------------------------------------------------

    keep_mask = ranked_scores >= min_similarity

    similar_indices = similar_indices[keep_mask]
    matched_scores = ranked_scores[keep_mask]

    return similar_indices, matched_scores



# ==========================================================
# STEP 8 : Main UI
# ==========================================================


st.markdown(
    """
    <div class="title">
    👕 StyleCluster AI
    </div>

    <div class="subtitle">
    AI-Based Fashion Recommendation System
    </div>

    """,
    unsafe_allow_html=True
)



uploaded_file = st.file_uploader(
    "📤 Upload Fashion Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file is not None:

    uploaded_image = Image.open(uploaded_file)

    st.markdown(
        """
        <div class="section">
        🖼 Uploaded Image
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col2:
        st.image(uploaded_image, width=350)

    with st.spinner("Finding similar fashion products..."):

        features = extract_features(uploaded_image)

        product_indices, product_scores = recommend_products(
            features,
            top_k=10
        )

    # No recommendation case: nothing cleared the similarity bar
    if len(product_indices) == 0:
        st.warning(
            "😕 No similar products found in our catalog. "
            "Try uploading a clearer fashion image."
        )
        st.stop()

    st.markdown(
        """
        <div class="section">
        ⭐ Recommended Similar Products
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display 5 images per row
    for start in range(0, len(product_indices), 5):

        cols = st.columns(5)

        row_products = product_indices[start:start + 5]
        row_scores = product_scores[start:start + 5]

        for col, index, score in zip(cols, row_products, row_scores):

            image_path = image_paths[index]

            with col:

                if os.path.exists(image_path):

                    img = Image.open(image_path)

                    st.image(img, use_container_width=True)
                    st.caption(f"Match: {score * 100:.1f}%")

                else:
                    st.warning(f"Missing file:\n{os.path.basename(image_path)}")



# ==========================================================
# Footer
# ==========================================================

st.markdown(
    """
    <br><br>

    <center>
    Powered by ResNet50 • PCA • Hierarchical Clustering
    </center>

    """,
    unsafe_allow_html=True
)