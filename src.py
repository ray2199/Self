# Data Models
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import uvicorn


class PopularityRecommender:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.data.dropna(inplace=True)

    def get_popular_products(self, top_n: int):
        popular_products = self.data.groupby("ProductId")["Rating"].count()
        most_popular = popular_products.sort_values(ascending=False).head(top_n)
        return most_popular.index.tolist()


class CollaborativeFilteringRecommender:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path).head(10000)
        self.utility_matrix = self._create_utility_matrix()
        self.decomposed_matrix = self._decompose_matrix()
        self.correlation_matrix = self._create_correlation_matrix()

    def _create_utility_matrix(self):
        return self.data.pivot_table(
            values="Rating", index="UserId", columns="ProductId", fill_value=0
        )

    def _decompose_matrix(self):
        svd = TruncatedSVD(n_components=10)
        return svd.fit_transform(self.utility_matrix.T)

    def _create_correlation_matrix(self):
        return np.corrcoef(self.decomposed_matrix)

    def get_recommendations(self, product_id: str, top_n: int):
        product_idx = list(self.utility_matrix.columns).index(product_id)
        correlation_product_ID = self.correlation_matrix[product_idx]
        recommended_ids = np.where(correlation_product_ID > 0.90)[0].tolist()
        recommended_ids.remove(product_idx)
        return [self.utility_matrix.columns[idx] for idx in recommended_ids][:top_n]


class ContentBasedRecommender:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path).dropna().head(500)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = self._fit_kmeans()

    def _fit_kmeans(self):
        X = self.vectorizer.fit_transform(self.data["product_description"])
        model = KMeans(n_clusters=10, init="k-means++")
        model.fit(X)
        return model

    def get_recommendations(self, query: str, top_n: int):
        Y = self.vectorizer.transform([query])
        prediction = self.model.predict(Y)
        cluster_products = self.data[self.model.labels_ == prediction[0]].head(top_n)
        return cluster_products["product_description"].tolist()
