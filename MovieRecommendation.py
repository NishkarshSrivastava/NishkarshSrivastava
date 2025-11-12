import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Load dataset
df = pd.read_csv("movies.csv")

# Convert genres to vector form
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = cv.fit_transform(df['genres'])

# Compute similarity
similarity = cosine_similarity(genre_matrix)
similarity_df = pd.DataFrame(similarity, index=df['title'], columns=df['title'])

def recommend(movie):
    if movie not in similarity_df.columns:
        return ["Movie not found."]
    recs = similarity_df[movie].sort_values(ascending=False)[1:6]
    return list(recs.index)

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")

movie_list = df['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader(f"Movies similar to '{selected_movie}':")
    for rec in recommendations:
        st.write(f"✅ {rec}")
