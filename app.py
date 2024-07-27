from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import uvicorn
from src import *

# FastAPI app
app = FastAPI()

popularity_recommender = PopularityRecommender("ratings_Beauty.csv")
collaborative_recommender = CollaborativeFilteringRecommender("ratings_Beauty.csv")
content_recommender = ContentBasedRecommender("product_descriptions.csv")


class QueryModel(BaseModel):
    product_id: str
    top_n: int = 10


@app.post("/recommend/popular")
def recommend_popular(top_n: int = 10):
    try:
        recommendations = popularity_recommender.get_popular_products(top_n)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/collaborative")
def recommend_collaborative(query: QueryModel):
    try:
        recommendations = collaborative_recommender.get_recommendations(
            query.product_id, query.top_n
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/content", response_class=HTMLResponse)
def recommend_content(
    query: str = Query(
        ..., description="Search query for content-based recommendations"
    ),
    top_n: int = 10,
):
    try:
        recommendations = content_recommender.get_recommendations(query, top_n)
        html_content = "<h2>Recommended Products</h2><ul>"
        for product in recommendations:
            html_content += f"<li>{product}</li>"
        html_content += "</ul>"
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serving static files for HTML UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
