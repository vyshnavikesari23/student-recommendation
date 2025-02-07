import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression

# Use mock API URLs
current_quiz_url = "http://localhost:5000/current_quiz"
historical_quiz_url = "http://localhost:5000/historical_quiz"

# Fetch Data
def fetch_data():
    current_quiz_data = requests.get(current_quiz_url).json()
    historical_quiz_data = requests.get(historical_quiz_url).json()
    return pd.DataFrame(current_quiz_data), pd.DataFrame(historical_quiz_data)

# Load Data
current_quiz_df, historical_quiz_df = fetch_data()

# Analyze Performance
def generate_insights(user_id):
    user_data = historical_quiz_df[historical_quiz_df["user_id"] == user_id]
    weak_topics = user_data.groupby("topic")["is_correct"].mean().sort_values().head(3).index.tolist()
    strong_topics = user_data.groupby("topic")["is_correct"].mean().sort_values(ascending=False).head(3).index.tolist()
    return {
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "average_score": user_data["score"].mean()
    }

# Performance by Topic
def analyze_performance_by_topic():
    topic_accuracy = historical_quiz_df.groupby("topic")["is_correct"].mean().reset_index()
    topic_accuracy.columns = ["Topic", "Accuracy"]
    print(topic_accuracy)

# Accuracy by Question Difficulty
def analyze_performance_by_difficulty():
    difficulty_accuracy = historical_quiz_df.groupby("difficulty")["is_correct"].mean().reset_index()
    difficulty_accuracy.columns = ["Difficulty Level", "Accuracy"]
    print(difficulty_accuracy)

# Score Trends Over Last 5 Quizzes
def plot_score_trend():
    historical_quiz_df.groupby("quiz_id")["score"].mean().plot(marker='o')
    plt.xlabel("Quiz ID")
    plt.ylabel("Average Score")
    plt.title("Performance Trend Over Last 5 Quizzes")
    plt.show()

# Generate Recommendations
def generate_recommendations(user_id):
    insights = generate_insights(user_id)
    recommendations = []
    if insights["average_score"] < 50:
        recommendations.append("Focus on revising fundamental concepts.")
    for topic in insights["weak_topics"]:
        recommendations.append(f"Practice more questions from {topic}.")
    return recommendations

# NEET Rank Prediction
def train_neet_model():
    neet_data = pd.DataFrame({
        "score": [100, 200, 300, 400, 500, 600, 650, 700],
        "rank": [500000, 400000, 300000, 200000, 100000, 50000, 20000, 1000]
    })
    X = np.array(neet_data["score"]).reshape(-1, 1)
    y = np.array(neet_data["rank"])
    model = LinearRegression()
    model.fit(X, y)
    return model

neet_model = train_neet_model()

def predict_neet_rank(user_score):
    predicted_rank = neet_model.predict(np.array([[user_score]]))
    return int(predicted_rank[0])

# Flask API
app = Flask(__name__)

@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    user_id = int(request.args.get("user_id"))
    recommendations = generate_recommendations(user_id)
    return jsonify({"user_id": user_id, "recommendations": recommendations})

@app.route("/predict_rank", methods=["GET"])
def get_predicted_rank():
    user_score = int(request.args.get("score"))
    rank = predict_neet_rank(user_score)
    return jsonify({"predicted_rank": rank})

if __name__ == "__main__":
    app.run(debug=True)
