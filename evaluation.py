from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form
from fastapi_utils.tasks import repeat_every
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score
import uvicorn

app = FastAPI()

# Define the location of the templates
templates = Jinja2Templates(directory="templates")


# Data Models
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

    def get_recommendations(self, user_id: str, top_n: int):
        if user_id not in self.utility_matrix.index:
            return []

        user_ratings = self.utility_matrix.loc[user_id]
        similar_users = self.correlation_matrix.dot(user_ratings) / np.array(
            [np.abs(self.correlation_matrix).sum(axis=1)]
        )

        similar_users = pd.Series(
            similar_users, index=self.utility_matrix.index
        ).sort_values(ascending=False)
        recommended_items = similar_users.head(top_n).index.tolist()

        return recommended_items

    def retrain(self, feedback_df: pd.DataFrame):
        for _, row in feedback_df.iterrows():
            user_id = row["user_id"]
            recommended_items = row["recommended_items"]
            feedback = row["feedback"]
            for item, rating in zip(recommended_items, feedback):
                self.utility_matrix.at[user_id, item] = rating

        self.decomposed_matrix = self._decompose_matrix()
        self.correlation_matrix = self._create_correlation_matrix()


class RecommendationEvaluator:
    def __init__(self, recommender, ground_truth_data_path: str, top_n: int = 10):
        self.recommender = recommender
        self.ground_truth = pd.read_csv(ground_truth_data_path)
        self.top_n = top_n

    def _get_recommendations(self, user_id):
        return self.recommender.get_recommendations(user_id, self.top_n)

    def evaluate(self):
        y_true = []
        y_pred = []

        for index, row in self.ground_truth.iterrows():
            user_id = row["UserId"]
            true_items = row["TrueItems"].split(",")
            recommended_items = self._get_recommendations(user_id)

            true_vector = [1 if item in true_items else 0 for item in recommended_items]
            pred_vector = [1] * len(recommended_items)

            y_true.extend(true_vector)
            y_pred.extend(pred_vector)

        precision = precision_score(y_true, y_pred, average="binary")
        recall = recall_score(y_true, y_pred, average="binary")

        return precision, recall


class FeedbackHandler:
    def __init__(self):
        self.feedback_data = []

    def log_feedback(self, user_id, recommended_items, feedback):
        self.feedback_data.append(
            {
                "user_id": user_id,
                "recommended_items": recommended_items,
                "feedback": feedback,
            }
        )

    def get_feedback(self):
        return pd.DataFrame(self.feedback_data)


feedback_handler = FeedbackHandler()
recommender = CollaborativeFilteringRecommender("ratings_Beauty.csv")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommend")
async def recommend(request: Request, user_id: str = Form(...), top_n: int = Form(...)):
    recommendations = recommender.get_recommendations(user_id, top_n)
    return templates.TemplateResponse(
        "index.html", {"request": request, "recommendations": recommendations}
    )


@app.post("/log_feedback")
async def log_feedback(
    request: Request,
    user_id: str = Form(...),
    recommended_items: str = Form(...),
    feedback: str = Form(...),
):
    recommended_items = recommended_items.split(",")
    feedback = list(map(int, feedback.split(",")))
    feedback_handler.log_feedback(user_id, recommended_items, feedback)
    return {"status": "feedback logged"}


@app.on_event("startup")
@repeat_every(seconds=3600)  # Adjust the interval as needed
def scheduled_evaluation():
    ground_truth_data_path = "ground_truth.csv"
    evaluator = RecommendationEvaluator(recommender, ground_truth_data_path)

    precision, recall = evaluator.evaluate()
    print(f"Evaluation - Precision: {precision}, Recall: {recall}")

    feedback_df = feedback_handler.get_feedback()
    if not feedback_df.empty:
        recommender.retrain(feedback_df)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
