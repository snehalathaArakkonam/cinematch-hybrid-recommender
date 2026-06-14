import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import os

DATA_DIR = "data"
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

svd = SVD(random_state=42)
svd.fit(trainset)
predictions = svd.test(testset)

rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

print("=== CineMatch Evaluation ===")
print(f"SVD RMSE: {rmse:.4f}")
print(f"SVD MAE: {mae:.4f}")
print("\nHybrid system ready for offline Precision@K evaluation.")