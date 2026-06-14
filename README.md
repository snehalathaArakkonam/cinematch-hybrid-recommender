# CineMatch — Hybrid Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

A modern **hybrid movie recommendation system** combining Content-Based Filtering and Collaborative Filtering using SVD.

## ✨ Features

- **Content-based**: TF-IDF + Cosine Similarity on genres
- **Collaborative**: Matrix Factorization with Surprise SVD
- **Hybrid Scoring**: Adjustable weight blending
- **FastAPI Backend** with clean REST endpoints
- **Single-file React Frontend** (beautiful dark UI)
- **Evaluation metrics** (RMSE, MAE, Precision@K ready)

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Frontend React] -->|HTTP| B[FastAPI Backend]
    B --> C[MovieLens Dataset]
    B --> D[TF-IDF Content Model]
    B --> E[SVD Collaborative Model]
    D & E --> F[Hybrid Engine]
    F --> A
```

## 🚀 Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
Open `frontend/index.html` in your browser.

## 📊 Evaluation

```bash
python evaluate.py
```

**SVD RMSE**: ~0.87 | **MAE**: ~0.67 (typical on MovieLens 100k)

## Endpoints
- `GET /recommend/{user_id}`
- `GET /similar/{movie_id}`
- `GET /movies/search?q=`

---

**Made with ❤️ for portfolio & learning**