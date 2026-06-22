import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

print("Loading datasets...")

# -----------------------
# LOAD RATINGS
# -----------------------

ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=["user_id","movie_id","rating","timestamp"]
)

# -----------------------
# LOAD MOVIES
# -----------------------

genre_cols = [
'unknown','Action','Adventure','Animation','Childrens',
'Comedy','Crime','Documentary','Drama','Fantasy',
'Film-Noir','Horror','Musical','Mystery',
'Romance','Sci-Fi','Thriller','War','Western'
]

movie_cols = [
'movie_id','title','release_date','video_release_date',
'imdb_url'
] + genre_cols

movies = pd.read_csv(
    "data/u.item",
    sep="|",
    encoding="latin-1",
    names=movie_cols
)

# -----------------------
# CONTENT-BASED MODEL
# -----------------------

print("Building TF-IDF model...")

movies["genres"] = movies[genre_cols].apply(
    lambda x: " ".join(
        [genre_cols[i] for i,val in enumerate(x) if val==1]
    ),
    axis=1
)

movies["content"] = movies["title"] + " " + movies["genres"]

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies["content"]
)

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

# -----------------------
# COLLABORATIVE FILTERING
# -----------------------

print("Training SVD model...")

reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(
    ratings[["user_id","movie_id","rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

svd = SVD()

svd.fit(trainset)

predictions = svd.test(testset)

print("\nEvaluation Results")

rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# -----------------------
# SAVE MODELS
# -----------------------

pickle.dump(
    svd,
    open("svd_model.pkl","wb")
)

pickle.dump(
    similarity_matrix,
    open("similarity.pkl","wb")
)

pickle.dump(
    movies,
    open("movies.pkl","wb")
)

pickle.dump(
    ratings,
    open("ratings.pkl","wb")
)

print("Training completed.")