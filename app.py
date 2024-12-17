from flask import Flask
import pickle
import numpy as np

from flask import Blueprint, render_template, request

model_path = "best_model.pkl"

with open(model_path, "rb") as f:
        model = pickle.load(f)

app = Flask(__name__)


def make_prediction(model, features):
    features = np.array([features])
    prediction = model.predict(features)[0]
    return prediction

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        print("Received Form Data:", form_data)  # Log form data for debugging

        # Parse inputs into a list for prediction
        features = [
            float(form_data.get("respiratory_rate")),
            float(form_data.get("c_reactive_proteins")),
            int(form_data.get("age")),
            float(form_data.get("tlc_count")),
        ]

        # Check feature length
        print("Parsed Features:", features)

        if len(features) != 4:
            return "Error: Incorrect number of features provided", 400

        # Get prediction
        prediction = make_prediction(model, features)

        if prediction == 0:
            res = "Not Severe"
        else:
            res = "Severe Condition"

        return render_template("result.html", prediction=res)

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500
