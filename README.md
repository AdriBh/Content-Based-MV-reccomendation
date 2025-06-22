# 🎬 Deep Learning-Based Content-Based Movie Recommendation System

This repository contains the code, model, and deployment for a content-based movie recommendation system built using deep learning techniques. The project leverages metadata from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and is deployed via a Streamlit web app.

👉 **Live Demo**: [Click here to try the app](https://kfgjbp9ssbs2sz6ztexxug.streamlit.app)

---

## 📌 Overview

This project demonstrates how autoencoder-based deep learning architectures can be used for recommending similar movies based on their content (e.g., plot overviews, genres, cast). It is ideal for situations where user ratings are sparse or unavailable.

### ✅ Features
- Deep learning encoder–decoder for learning movie embeddings
- TF-IDF vectorization of combined metadata and descriptions
- Cosine similarity-based recommendation engine
- Real-time recommendations in a deployed Streamlit app
- Fully content-based (no user interaction required)

---

## 🧠 Architecture

1. **Data Preprocessing**:
   - Merged metadata: title, overview, genres, keywords, cast, and crew
   - Cleaned and tokenized text fields
   - Applied TF-IDF vectorization

2. **Model**:
   - Autoencoder model in Keras/TensorFlow
   - Latent embedding layer trained to compress TF-IDF vectors

3. **Similarity Engine**:
   - Generated embeddings used to compute cosine similarity between movies
   - Top-N similar movies retrieved for a given input

4. **Frontend**:
   - Interactive movie selection UI using Streamlit
   - Posters, descriptions, and similar movies displayed

---

## 🧰 Tools & Technologies
Python (3.8+)

 -TensorFlow / Keras – for autoencoder model

 -scikit-learn – TF-IDF, preprocessing, cosine similarity

 -pandas, NumPy – data wrangling

 -Streamlit – app deployment and interface
