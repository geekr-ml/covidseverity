from flask import Flask
import pickle
import numpy as np

from flask import Blueprint, render_template, request

model_path = "final_model_for_TBTO.pkl"

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
            int(form_data.get("types_of_cases")),
            int(form_data.get("site_of_tb_disease")),
            int(form_data.get("hiv_status")),
            int(form_data.get("bacteriologically_confirmed")),
            int(form_data.get("rif_resistance_detected")),
            int(form_data.get("rif_resistance_not_detected")),
            int(form_data.get("rr_tb")),
            float(form_data.get("age")),
            float(form_data.get("height")),
            float(form_data.get("weight")),
            int(form_data.get("days_in_treatment"))
        ]

        # Check feature length
        print("Parsed Features:", features)

        if len(features) != 11:
            return "Error: Incorrect number of features provided", 400

        # Get prediction
        prediction = make_prediction(model, features)

        if prediction == 0:
            res = "Cured"
        elif prediction == 1:
            res = "Died" 
        elif prediction == 2:
            res = "LOST_TO_FOLLOW_UP"
        elif prediction == 3:
            res = "TREATMENT_COMPLETE"
        else:
            res = "Treatment Success"

        return render_template("result.html", prediction=res)

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)