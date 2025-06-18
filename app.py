import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model


# --- Unzip df_c.pkl from data_file.zip if not already extracted ---
if not os.path.exists("df_c.pkl"):
    with zipfile.ZipFile("data_file.zip", "r") as zip_ref:
        zip_ref.extract("df_c.pkl")
if not os.path.exists("features.pkl"):
    with zipfile.ZipFile("features.zip", "r") as zip_ref1:
        zip_ref1.extract("features.pkl")

# --- Load Data ---
@st.cache_resource
def load_all_data():
    features = pd.read_pickle("features.pkl")
    with open("genre_columns.pkl", "rb") as f:
        genre_columns = pickle.load(f)
    model = load_model("autoencoder_model.keras")
    df_c = pd.read_pickle("df_c.pkl")
    movie_embeddings = model.predict(features.values)
    return features, df_c, genre_columns, model, movie_embeddings

features, df_c, genre_columns, model, movie_embeddings = load_all_data()

def get_recommendations(liked_titles, liked_genres, top_n=10, movie_weight=0.7, genre_weight=0.3):
    # --- Movie vector ---
    liked_movie_vectors = features[df_c['original_title'].isin(liked_titles)]
    if not liked_movie_vectors.empty:
        liked_movie_mean = liked_movie_vectors.mean()
    else:
        liked_movie_mean = pd.Series([0.0] * features.shape[1], index=features.columns)
        movie_weight = 0.0  # Ignore movie contribution if no titles

    # --- Genre vector ---
    genre_vector = pd.Series([0.0] * len(genre_columns), index=genre_columns)
    for g in liked_genres:
        if g in genre_vector:
            genre_vector[g] = 1.0

    genre_vector_full = pd.Series([0.0] * features.shape[1], index=features.columns)
    genre_vector_full.update(genre_vector)

    if genre_vector.sum() == 0:
        genre_weight = 0.0  # Ignore genre contribution if none selected

    # --- Combine weighted profile ---
    if movie_weight + genre_weight == 0:
        return pd.DataFrame(columns=['original_title', 'score'])  # No input case

    # Normalize weights to sum to 1
    total_weight = movie_weight + genre_weight
    movie_weight /= total_weight
    genre_weight /= total_weight

    user_profile_series = movie_weight * liked_movie_mean + genre_weight * genre_vector_full
    user_profile = user_profile_series.values.reshape(1, -1)

    # --- Compute embedding & similarity ---
    user_embedding = model.predict(user_profile)
    sim_scores = cosine_similarity(user_embedding, movie_embeddings).flatten()

    df_copy = df_c.copy()
    df_copy['score'] = sim_scores
    recommendations = df_copy.sort_values(by='score', ascending=False)
    return recommendations[['original_title', 'score']].head(top_n)


# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Personalized Movie Recommender")

all_titles = sorted(df_c['original_title'].dropna().unique())
liked_titles = st.multiselect("🎞️ Select movies you loved watching:", all_titles)

available_genres = sorted(genre_columns)
liked_genres = st.multiselect("🎭 Select your favorite genres:", available_genres)

if st.button("🎯 Recommend"):
    if not liked_titles and not liked_genres:
        st.warning("Please select at least one movie or genre.")
    else:
        with st.spinner("🔍 Finding the best matches for you..."):
            recs = get_recommendations(liked_titles, liked_genres, top_n=10)
        st.success("✅ Top Recommendations:")
        st.table(recs)
