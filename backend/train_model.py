# train_model.py
# Run this file ONCE to train the model and save it as model.pkl

import pandas as pd                                   # pandas: reads and handles table data (CSV)
from sklearn.model_selection import train_test_split  # splits data into training and testing parts
from sklearn.tree import DecisionTreeClassifier, export_text  # the decision tree + a text printer
from sklearn.svm import SVC                           # SVM (only used for comparison)
from sklearn.preprocessing import StandardScaler      # puts all numbers on a similar scale (SVM needs this)
from sklearn.metrics import accuracy_score            # calculates accuracy
import joblib                                         # saves the trained model to a file

# ---------- 1. LOAD THE DATASET ----------
data = pd.read_csv("loan_data.csv")                   # read the CSV into a table called "data"
print("Rows in dataset:", len(data))

# ---------- 2. PREPROCESSING ----------
# A model understands only numbers, so text columns are converted to numbers.
# These two dictionaries are the "translation tables".
employment_map = {"Unemployed": 0, "Self-Employed": 1, "Salaried": 2}
repayment_map = {"Defaulted": 0, "No History": 1, "Repaid": 2}

data["employment_status"] = data["employment_status"].map(employment_map)
data["previous_repayment"] = data["previous_repayment"].map(repayment_map)

# ---------- 3. SEPARATE FEATURES (X) AND TARGET (y) ----------
feature_names = ["age", "monthly_income", "employment_status", "loan_amount",
                 "loan_term", "credit_score", "existing_loans",
                 "debt_to_income", "previous_repayment"]
X = data[feature_names].values    # X = the inputs (9 columns)
y = data["risk"]                  # y = the answer we want to predict

# ---------- 4. TRAIN / TEST SPLIT ----------
# 75% rows for learning, 25% rows hidden from the model for testing.
# random_state=42 makes the split the same every time we run the file.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print("Training rows:", len(X_train), " Testing rows:", len(X_test))

# ---------- 5. TRAIN THE ID3-STYLE DECISION TREE ----------
# criterion="entropy" -> the tree picks splits using Entropy and Information Gain (the ID3 idea)
# max_depth=5         -> tree can have at most 5 questions in a row (keeps it small, avoids overfitting)
# min_samples_leaf=2  -> every final box (leaf) must contain at least 2 training rows
tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                    min_samples_leaf=2, random_state=42)
tree_model.fit(X_train, y_train)                      # fit = learn from the training data

# ---------- 6. TEST THE MODEL ----------
tree_predictions = tree_model.predict(X_test)         # predict on rows the model has never seen
tree_accuracy = accuracy_score(y_test, tree_predictions)
print("Decision Tree accuracy:", round(tree_accuracy * 100, 1), "%")

# ---------- 7. (OPTIONAL) SVM FOR COMPARISON ONLY ----------
# SVM is sensitive to the size of numbers (income ~50000 vs age ~30), so we scale first.
# A decision tree does NOT need this step - another reason trees are simpler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model = SVC(kernel="linear")
svm_model.fit(X_train_scaled, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))
print("SVM accuracy (comparison only):", round(svm_accuracy * 100, 1), "%")

# ---------- 8. PRINT THE TREE AS TEXT (nice for your viva) ----------
print("\nLearned tree rules:")
print(export_text(tree_model, feature_names=feature_names))

# ---------- 9. SAVE THE TRAINED TREE ----------
joblib.dump(tree_model, "model.pkl")                  # Flask will load this file later
print("Model saved as model.pkl")
