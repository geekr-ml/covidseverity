from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model_path = "best_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Helper function to make predictions
def make_prediction(model, features):
    features = np.array([features])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities

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

        # Make prediction and get probabilities
        prediction, probabilities = make_prediction(model, features)
        probability_positive = probabilities[1]
        probability_negative = probabilities[0]

        # Map prediction result to human-readable labels
        res = "Severe Condition" if prediction == 1 else "Not Severe"

        # Prepare entered values for display in the result template
        entered_values = {
            "Respiratory Rate": features[0],
            "C-Reactive Proteins": features[1],
            "Age": features[2],
            "TLC Count": features[3],
        }

        return render_template(
            "result.html",
            prediction=res,
            probability_positive=round(probability_positive * 100, 2),
            probability_negative=round(probability_negative * 100, 2),
            entered_values=entered_values,
        )

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
