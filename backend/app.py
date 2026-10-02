# app.py - a very small Flask API for the Loan Risk project

from flask import Flask, request, jsonify   # Flask = web framework, request = incoming data, jsonify = make JSON reply
from flask_cors import CORS                 # allows React (port 5173) to call Flask (port 5000)
import joblib                               # loads the saved model

app = Flask(__name__)    # create the Flask application
CORS(app)                # switch on CORS for the whole app

# ---------- LOAD THE TRAINED MODEL ----------
try:
    model = joblib.load("model.pkl")        # file created by train_model.py
except FileNotFoundError:
    model = None                            # remember that the file is missing

# ---------- LISTS THAT MUST MATCH train_model.py ----------
# The model expects the 9 values in exactly this order.
feature_names = ["age", "monthly_income", "employment_status", "loan_amount",
                 "loan_term", "credit_score", "existing_loans",
                 "debt_to_income", "previous_repayment"]

# Text -> number tables (same as in train_model.py)
employment_map = {"Unemployed": 0, "Self-Employed": 1, "Salaried": 2}
repayment_map = {"Defaulted": 0, "No History": 1, "Repaid": 2}

# Friendly names used when we write the explanation
labels = {"age": "Age", "monthly_income": "Monthly Income (Rs)",
          "employment_status": "Employment Status", "loan_amount": "Loan Amount (Rs)",
          "loan_term": "Loan Term (months)", "credit_score": "Credit Score",
          "existing_loans": "Existing Loans", "debt_to_income": "Debt-to-Income Ratio (%)",
          "previous_repayment": "Previous Loan Repayment"}

# Sentence to show when the tree went LEFT (value <= threshold) or RIGHT (value > threshold)
reason_text = {"age": ("Age is on the lower side", "Age is on the higher side"),
               "monthly_income": ("Monthly income is low", "Monthly income is sufficient"),
               "loan_amount": ("Loan amount is small", "Loan amount is large"),
               "loan_term": ("Loan term is short", "Loan term is long"),
               "credit_score": ("Credit score is low", "Credit score is high"),
               "existing_loans": ("Number of existing loans is low", "Number of existing loans is high"),
               "debt_to_income": ("Debt-to-income ratio is low", "Debt-to-income ratio is high")}


# A simple GET route, just to check that the server is running
@app.route("/", methods=["GET"])
def home():
    return "Loan Risk API is running"


# The main route: React sends applicant data here using POST
@app.route("/predict", methods=["POST"])
def predict():
    # --- 1. Check the model is available ---
    if model is None:
        return jsonify({"error": "model.pkl not found. Run train_model.py first."}), 500

    # --- 2. Read the JSON sent by React ---
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No data received"}), 400

    # --- 3. Check that no field is missing or empty ---
    for name in feature_names:
        if name not in data or str(data[name]).strip() == "":
            return jsonify({"error": "Please fill in: " + labels[name]}), 400

    # --- 4. Convert text to numbers ---
    if data["employment_status"] not in employment_map or data["previous_repayment"] not in repayment_map:
        return jsonify({"error": "Invalid employment status or repayment value"}), 400

    try:
        age = float(data["age"])
        income = float(data["monthly_income"])
        loan_amount = float(data["loan_amount"])
        loan_term = float(data["loan_term"])
        credit_score = float(data["credit_score"])
        existing_loans = float(data["existing_loans"])
        debt_to_income = float(data["debt_to_income"])
    except ValueError:
        return jsonify({"error": "Numeric fields must contain numbers only"}), 400

    # Simple range checks
    if age < 18 or credit_score < 300 or credit_score > 900 or debt_to_income < 0 or debt_to_income > 100:
        return jsonify({"error": "Check age (18+), credit score (300-900) and debt-to-income (0-100)"}), 400

    employment_number = employment_map[data["employment_status"]]
    repayment_number = repayment_map[data["previous_repayment"]]

    # --- 5. Build the row in the same order used during training ---
    applicant = [[age, income, employment_number, loan_amount, loan_term,
                  credit_score, existing_loans, debt_to_income, repayment_number]]

    # --- 6. Ask the model for the prediction and confidence ---
    risk = model.predict(applicant)[0]                     # e.g. "Low Risk"
    confidence = float(max(model.predict_proba(applicant)[0]))   # highest class probability

    # --- 7. Build the explanation from the tree's REAL path ---
    tree = model.tree_                                     # the internal tree structure
    node_ids = model.decision_path(applicant).indices      # numbers of the boxes (nodes) this applicant passed through

    decision_path = []     # steps like "Credit Score > 595"
    reasons = []           # sentences like "Credit score is high (750)"

    for node_id in node_ids:
        if tree.children_left[node_id] == -1:              # -1 means this node is a leaf (final answer), so skip it
            continue
        feature_index = tree.feature[node_id]              # which column this node asked about
        feature_name = feature_names[feature_index]
        threshold = tree.threshold[node_id]                # the number it compared against
        value = applicant[0][feature_index]                # this applicant's value for that column
        label = labels[feature_name]

        if feature_name == "employment_status":            # text columns: show the original word
            decision_path.append(label + " = " + data["employment_status"])
            reasons.append("Employment status is " + data["employment_status"])
        elif feature_name == "previous_repayment":
            decision_path.append(label + " = " + data["previous_repayment"])
            reasons.append("Previous loan repayment: " + data["previous_repayment"])
        elif value <= threshold:                           # tree went LEFT
            decision_path.append(label + " <= " + format(threshold, "g"))
            reasons.append(reason_text[feature_name][0] + " (" + format(value, "g") + " <= " + format(threshold, "g") + ")")
        else:                                              # tree went RIGHT
            decision_path.append(label + " > " + format(threshold, "g"))
            reasons.append(reason_text[feature_name][1] + " (" + format(value, "g") + " > " + format(threshold, "g") + ")")

    decision_path.append("Prediction = " + risk)

    # --- 8. Send everything back to React as JSON ---
    return jsonify({"risk": risk, "confidence": round(confidence, 2),
                    "reasons": reasons, "decision_path": decision_path})


if __name__ == "__main__":
    app.run(debug=True)    # start the server on http://127.0.0.1:5000
