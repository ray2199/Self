# Product Recommendation System

This project is a product recommendation system built using FastAPI. It provides recommendations based on collaborative filtering and allows users to provide feedback to improve the recommendation system over time.

## Features

- Get product recommendations for a user.
- Provide feedback on the recommended products.
- Periodically evaluate and retrain the recommendation system based on user feedback.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn
- Jinja2
- FastAPI-utils

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ray2199/Self.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. Ensure you have the required CSV files (`ratings_Beauty.csv` and `ground_truth.csv`) in the project directory.

2. Start the FastAPI server by running the app:

3. Open your web browser and navigate to `http://127.0.0.1:8000` to access the application.
