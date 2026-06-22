import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
# SIDEBAR
# -----------------------

st.sidebar.title("Project Information")

st.sidebar.write("""
Movie Recommendation System
Algorithms Used:
- TF-IDF
- Cosine Similarity
- SVD
- Hybrid Recommendation
Dataset:
MovieLens 100K
""")

st.sidebar.subheader("Evaluation Metrics")

st.sidebar.metric("RMSE", "0.93")
st.sidebar.metric("MAE", "0.73")
st.sidebar.metric("Precision@5", "0.72")
st.sidebar.metric("Recall@5", "0.65")
st.sidebar.metric("Coverage", "63.2%")

# -----------------------
# STREAMLIT UI
# -----------------------

st.title(
    "Movie Recommendation System"
)

st.write(
    "Get personalized movie recommendations using a hybrid recommendation engine."
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=943,
    value=1
)

if st.button("Recommend"):

    with st.spinner("Generating recommendations... "):
        user_ratings = ratings[
            ratings["user_id"] == user_id
                               ]
        
        st.write(f"user{user_id} has rated {len(user_ratings)} movies. ")


    # st.write("Generating recommendations... ")

    results = recommend_movies(user_id)

    # st.write("finished")

    st.subheader("Top Recommendations")

    result_df = pd.DataFrame(
    results,
    columns=["Movie", "Hybrid Score"]
    )

    st.dataframe(result_df)

    titles = [x[0] for x in results]
    scores = [x[1] for x in results]

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        titles[::-1],
        scores[::-1]
    )

    ax.set_title(
        "Top Recommended Movies"
    )

    ax.set_xlabel(
        "Hybrid Score"
    )

    ax.set_ylabel(
        "Movies"
    )
    st.pyplot(fig)

    # for title, score in results:

    #     st.write(
    #         f"{title}  (Score: {score:.2f})"
    #     )