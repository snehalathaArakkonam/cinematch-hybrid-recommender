from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np
import os
from typing import List, Dict

app = FastAPI(title="CineMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
DATA_DIR = "../data"
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# Content-based
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(random_state=42)
svd.fit(trainset)

print("✅ Models trained successfully!")

@app.get("/movies/search")
def search_movies(q: str):
    if not q or len(q) < 2:
        return []
    results = movies[movies['title'].str.contains(q, case=False, na=False)].head(20).to_dict('records')
    return results

@app.get("/similar/{movie_id}")
def get_similar_movies(movie_id: int):
    try:
        idx = movies[movies['movieId'] == movie_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices].to_dict('records')
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, alpha: float = 0.5):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        popular = ratings.groupby('movieId').size().nlargest(50).index.tolist()
        recs = movies[movies['movieId'].isin(popular[:10])].to_dict('records')
        return {"recommendations": recs, "hybrid_scores": [0.0]*len(recs)}
    
    user_movie_ids = user_ratings['movieId'].tolist()
    user_indices = movies[movies['movieId'].isin(user_movie_ids)].index.tolist()
    if not user_indices:
        user_indices = [0]
    
    recommendations = []
    seen = set(user_movie_ids)
    
    for idx in range(len(movies)):
        movie_id = int(movies.iloc[idx]['movieId'])
        if movie_id in seen:
            continue
        content_score = max([cosine_sim[idx][u_idx] for u_idx in user_indices]) if user_indices else 0.0
        try:
            collab_pred = svd.predict(user_id, movie_id).est
        except:
            collab_pred = 3.0
        hybrid_score = alpha * content_score + (1 - alpha) * (collab_pred / 5.0)
        recommendations.append((movie_id, hybrid_score, collab_pred))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recs = recommendations[:10]
    
    rec_movies = []
    hybrid_scores = []
    for mid, hscore, cscore in top_recs:
        movie = movies[movies['movieId'] == mid].iloc[0].to_dict()
        movie['predicted_rating'] = round(cscore, 2)
        rec_movies.append(movie)
        hybrid_scores.append(round(hscore, 3))
    
    return {"recommendations": rec_movies, "hybrid_scores": hybrid_scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)