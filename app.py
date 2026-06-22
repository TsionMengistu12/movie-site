import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------
# LOAD FILES
# -----------------------

svd = pickle.load(open("svd_model.pkl","rb"))

similarity = pickle.load(
    open("similarity.pkl","rb")
)

movies = pickle.load(
    open("movies.pkl","rb")
)

ratings = pickle.load(
    open("ratings.pkl","rb")
)

# -----------------------
# RECOMMEND FUNCTION
# -----------------------

def recommend_movies(user_id, top_n=10):

    rated_movies = ratings[
        ratings["user_id"] == user_id
    ]["movie_id"].tolist()

    recommendations = []

    for movie_id in movies["movie_id"][:300]:

        if movie_id in rated_movies:
            continue

        # Collaborative score
        svd_score = svd.predict(
            user_id,
            movie_id
        ).est

        # Content score
        content_score = 0

        for rated_movie in rated_movies:

            try:

                idx1 = movies[
                    movies["movie_id"]==movie_id
                ].index[0]

                idx2 = movies[
                    movies["movie_id"]==rated_movie
                ].index[0]

                content_score += similarity[idx1][idx2]

            except:
                pass

        if len(rated_movies) > 0:
            content_score /= len(rated_movies)

        hybrid_score = (
            0.7 * svd_score
            +
            0.3 * content_score
        )

        title = movies[
            movies["movie_id"]==movie_id
        ]["title"].values[0]

        recommendations.append(
            (title, hybrid_score)
        )

    recommendations.sort(
        key=lambda x:x[1],
        reverse=True
    )

    return recommendations[:top_n]

# -----------------------
# STREAMLIT UI
# -----------------------

st.title(
    "Hybrid Movie Recommendation System"
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=943,
    value=1
)

if st.button("Recommend"):
    st.write("Generating recommendations... ")

    results = recommend_movies(user_id)

    st.write("finished")

    st.subheader("Top Recommendations")

    for title, score in results:

        st.write(
            f"{title}  (Score: {score:.2f})"
        )