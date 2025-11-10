# app.py
from flask import Flask, request, render_template, jsonify
import joblib
from utils import clean_text
import os

app = Flask(__name__)
MODEL_PATH = os.path.join("models", "spam_model.joblib")

model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or request.form
    text = data.get("text","")
    if not text:
        return jsonify({"error":"No text provided"}), 400
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    prob = float(model.predict_proba([cleaned])[0][1])  # prob of spam
    label = "spam" if pred==1 else "ham"
    return jsonify({"label": label, "spam_prob": round(prob, 4)})

if __name__ == "__main__":
    app.run(debug=True)
