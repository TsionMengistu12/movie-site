from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

ratings = pd.read_csv(
    "../data/u.data",
    sep="\t",
    names=["user_id","movie_id","rating","timestamp"]
)

reader = Reader(rating_scale=(1,5))

data = Dataset.load_from_df(
    ratings[['user_id','movie_id','rating']],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

print("RMSE")
accuracy.rmse(predictions)

print("MAE")
accuracy.mae(predictions)