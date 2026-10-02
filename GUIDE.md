# Explainable Loan Risk Prediction System — Complete Guide

## Folder structure

```
loan-risk-project/
├── backend/
│   ├── app.py              <- Flask API (/predict)
│   ├── train_model.py      <- trains and saves the model
│   ├── loan_data.csv       <- 100-row dataset
│   ├── requirements.txt    <- Python libraries needed
│   └── model.pkl           <- CREATED when you run train_model.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js      <- one extra file React (Vite) needs
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── App.css
```

I added only one file beyond your suggested structure: `vite.config.js`. Vite (the tool that runs React in development) needs it to know we use React. You never edit it.

> **Tested:** I trained the model, called `/predict` with real requests, and built the React app. All outputs in this guide are real outputs, not invented.

---

# RUNNING INSTRUCTIONS (Windows + VS Code)

### Step 0 — Install two programs
1. **Python 3.10+** from python.org. On the first installer screen, **tick "Add python.exe to PATH"**.
2. **Node.js (LTS)** from nodejs.org. React needs it (it also installs `npm`).

Open VS Code → *Terminal → New Terminal* (PowerShell). Check:
```powershell
python --version
node --version
npm --version
```
If `python` is not found, try `py --version`, or reinstall Python and tick the PATH box.

### Step 1 — Put the project in a folder
Copy the `loan-risk-project` folder anywhere (e.g. Desktop), then in VS Code: *File → Open Folder → loan-risk-project*.

### Step 2 — Backend setup (Terminal 1)
```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
```
A virtual environment (`venv`) is a private box of Python libraries just for this project, so it does not disturb other projects. You will see `(venv)` at the start of the prompt.

If PowerShell says *"running scripts is disabled"*, run this once, then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install the libraries:
```powershell
pip install flask flask-cors pandas scikit-learn joblib
```
(or simply `pip install -r requirements.txt` — same thing). What each one is: `flask` = web server, `flask-cors` = allows React to call Flask, `pandas` = reads the CSV, `scikit-learn` = the decision tree, `joblib` = saves/loads the model.

### Step 3 — Train the model
```powershell
python train_model.py
```
You should see the accuracy and the learned tree, then `Model saved as model.pkl`. **Run this on your own computer** (do not copy a `model.pkl` from someone else — saved models can break between scikit-learn versions).

### Step 4 — Start Flask
```powershell
python app.py
```
Leave this terminal running. Open http://127.0.0.1:5000 in a browser; it should say **Loan Risk API is running**.

### Step 5 — Start React (Terminal 2)
Click the **+** in the VS Code terminal panel to open a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
`npm install` reads `package.json` and downloads React and Vite into a `node_modules` folder (takes a minute, needs internet). `npm run dev` starts the website.

### Step 6 — URLs
| What | URL |
|---|---|
| The website (open this) | http://localhost:5173 |
| Flask API check | http://127.0.0.1:5000 |

### How React is connected to Flask
There is no special setup. `App.jsx` calls `fetch("http://127.0.0.1:5000/predict", ...)`, and Flask has `CORS(app)` so the browser allows it. Both servers must be running at the same time.

> If you prefer to create React yourself: `npm create vite@latest frontend -- --template react`, then replace `src/App.jsx`, `src/App.css`, `src/main.jsx` with mine.

---

# PART 1 — Project architecture

```
User types in the form (React, port 5173)
        ↓
React sends an HTTP POST request with JSON  →  http://127.0.0.1:5000/predict
        ↓
Flask (Python, port 5000) receives the JSON
        ↓
Flask converts the data into numbers and gives them to the ML model (model.pkl)
        ↓
Model returns a prediction; Flask reads the tree's path to build the explanation
        ↓
Flask sends back JSON: risk, confidence, reasons, decision_path
        ↓
React stores the JSON in state and displays it
```

| Part | Job |
|---|---|
| **React** | The screen: form, button, showing results. Runs in the browser (JavaScript). |
| **Flask** | A small Python program that waits for requests and replies. It is the "bridge" between the browser and Python. |
| **ML model** | The trained decision tree. Gets 9 numbers, returns Low/Medium/High. |
| **Explanation** | Built in Flask by following the exact boxes (nodes) the applicant passed through in the tree. |

**Why can't React run the Python model directly?** React runs in the browser, which only understands JavaScript. scikit-learn is Python. So we need a Python server in between.

---

# PART 2 — Machine Learning

## 2.1 The dataset (`loan_data.csv`)
100 rows, one row = one past loan applicant. **Important to say honestly in your viva: the data is synthetic (made up for learning)**, created with simple rules (e.g. a defaulted applicant or very low credit score = High Risk). Real bank data is private and huge.

| Column | Meaning | Example |
|---|---|---|
| `age` | Applicant age in years | 34 |
| `monthly_income` | Income per month in Rs | 95000 |
| `employment_status` | Salaried / Self-Employed / Unemployed | Salaried |
| `loan_amount` | Loan requested in Rs | 100000 |
| `loan_term` | Repayment period in months | 12 |
| `credit_score` | Score 300–900 (higher = better) | 750 |
| `existing_loans` | Number of loans already running | 3 |
| `debt_to_income` | % of income already going to debt payments | 10 |
| `previous_repayment` | Repaid / No History / Defaulted | Repaid |
| **`risk`** | **TARGET: Low Risk / Medium Risk / High Risk** | Medium Risk |

- **Features (X)** = the 9 input columns. **Target (y)** = the `risk` column, the answer we want to predict.

## 2.2 Preprocessing
Models only understand numbers. Two columns contain words, so we convert them with dictionaries:

```
employment_status:  Unemployed=0, Self-Employed=1, Salaried=2
previous_repayment: Defaulted=0,  No History=1,    Repaid=2
```
(This is called *ordinal encoding*: the numbers are in a meaningful order — bigger is "better".) A decision tree does **not** need scaling of numeric columns (income 60000 vs age 25 is fine). SVM does, which is why `train_model.py` scales only for the SVM.

## 2.3 Train/test split
- 75 rows (75%) are for **training**: the model learns from them.
- 25 rows (25%) are **testing**: kept hidden, used afterwards to check if the model works on new data.
- `random_state=42` just fixes the random shuffle so you get the same result every time.

Why split? If we tested on the same rows we trained on, the model could just memorise them and the score would be meaningless.

## 2.4 ID3, Entropy, Information Gain

**Decision tree** = a flowchart of yes/no questions. Each box asks about one feature ("Credit score > 595?"), and the leaves give the answer.

**Entropy** = how mixed/impure a group is.
- 10 applicants, all High Risk → totally pure → **entropy = 0**.
- 10 applicants, 5 Low and 5 High → maximum mixing (for 2 classes) → **entropy = 1**.

Formula: `Entropy = − Σ p × log2(p)` where p = fraction of each class.

**Information Gain** = entropy before the split − weighted average entropy after the split. The attribute with the **highest** gain is chosen for the question. ID3 repeats this on every branch until the groups are pure (or a stopping rule hits).

**Real numbers from this project's training data** (root of the tree):
- 75 training rows: 27 High, 24 Medium, 24 Low → entropy ≈ **1.583** (very mixed; 3 classes so max is 1.585).
- Question tried: `credit_score <= 595`
  - Yes-side: 19 rows, **all High Risk** → entropy 0
  - No-side: 56 rows (24 Medium, 24 Low, 8 High) → entropy ≈ 1.449
  - Weighted after-split entropy = (19/75)×0 + (56/75)×1.449 ≈ 1.082
  - **Information Gain = 1.583 − 1.082 ≈ 0.50**

That was the best gain of all questions, so it became the root. This is exactly why the tree's first question is the credit score.

## 2.5 Is scikit-learn's tree really ID3? (be honest in the viva)
scikit-learn has **no pure ID3**. Its `DecisionTreeClassifier` is an optimised **CART**. Setting `criterion="entropy"` makes it choose splits by **entropy / information gain, which is the ID3 idea**. So we say: *"We use the ID3 approach — entropy and information gain — implemented through scikit-learn's DecisionTreeClassifier."* Differences:

| | ID3 | CART (what scikit-learn builds) |
|---|---|---|
| Split measure | Entropy / Information gain | Gini (default) or entropy |
| Branches per question | Many (one per category value) | Always 2 (yes/no) |
| Numeric columns | Needs them turned into categories | Handles directly with thresholds like `<= 595` |
| Output | Classification only | Classification and regression |

## 2.6 SVM (comparison only)
SVM draws the best separating boundary (hyperplane) between classes, with the widest possible margin. It can be accurate but is a "black box": you cannot say "it chose High because income was low" in simple words. In our run: Decision Tree **92%**, SVM **84%** on the same test rows (your numbers may differ slightly if you change the data). So the tree was better *and* explainable.

## 2.7 Prediction and accuracy
To predict, a new applicant walks from the top box downwards, answering each question, until a leaf. The leaf's majority class is the prediction. **Confidence** = share of training rows in that leaf that had that class (0.96 means 96%).

**Accuracy** = correct predictions ÷ total test rows. Ours: 23/25 = **92%**.

## 2.8 The learned tree (real output)
```
credit_score <= 595            -> High Risk
credit_score >  595
  previous_repayment = Defaulted          -> High Risk
  previous_repayment = No History
      monthly_income <= 27500 -> High Risk, else Medium Risk
  previous_repayment = Repaid
      existing_loans <= 2.5
          monthly_income <= 40000 -> (Unemployed: High, else Medium)
          monthly_income >  40000
              debt_to_income <= 35 -> LOW RISK
              debt_to_income >  35 -> Medium Risk
      existing_loans > 2.5
          credit_score <= 810 -> Medium Risk, else High Risk
```
Note: the tree is **learned from data**, not written by hand. It will not match your example ("Credit Score > 700 → Income > 50000 ...") exactly; it uses whatever splits had the best information gain. The odd last rule (credit > 810 and many loans → High) is a small-data quirk — a good example to mention for **overfitting**.

---

# PART 3 — Flask

## 3.1 Basics
- **Flask** = a small Python library that turns a Python file into a web server.
- **Why needed?** The browser (React) can't run Python/scikit-learn, so Flask runs the model and answers requests.
- **API** = a set of "doors" (URLs) that programs use to talk to each other. React asks, Flask answers. No web page is returned, only data.
- **`/predict`** = the address (route) of the door that does predictions. Full address: `http://127.0.0.1:5000/predict`. `127.0.0.1` means "this same computer", `5000` is the port Flask listens on.
- **GET vs POST**
  - GET = "give me something" (opening a page). Data would go in the URL.
  - POST = "here is some data, process it". Data goes in the request body. We use POST because we send a whole form.
- **JSON** = text format for data that both JavaScript and Python understand: `{"age": 25, "credit_score": 750}`. Looks like a JS object / Python dictionary.
- **`request`** = Flask object holding what the browser sent. `request.get_json()` reads the JSON body into a Python dictionary.
- **`jsonify`** = turns a Python dictionary into a JSON reply (with the correct headers).
- **CORS** = browser safety rule. A page at `localhost:5173` is not allowed to read replies from `127.0.0.1:5000` (a different "origin") unless that server says OK. `CORS(app)` makes Flask say OK. Without it you get a "blocked by CORS policy" error in the browser console, even though Flask works.

## 3.2 `app.py` line-by-line

**Imports and setup**
| Code | Meaning |
|---|---|
| `from flask import Flask, request, jsonify` | Bring in the three Flask tools we use |
| `from flask_cors import CORS` | Bring in CORS support |
| `import joblib` | Tool to load the saved model |
| `app = Flask(__name__)` | Creates the web app. `__name__` just tells Flask which file it is in |
| `CORS(app)` | Allows React to call this server |

**Loading the model**
| Code | Meaning |
|---|---|
| `try: model = joblib.load("model.pkl")` | Load the trained tree from file into variable `model` |
| `except FileNotFoundError: model = None` | If the file does not exist, don't crash; remember with `None` and report an error later |

**Lookup tables**
| Code | Meaning |
|---|---|
| `feature_names = [...]` | The 9 columns **in training order**. Order must match or predictions are wrong |
| `employment_map`, `repayment_map` | Same word→number tables as in training |
| `labels` | Nice names for showing in explanation text |
| `reason_text` | For each numeric feature: (sentence if value is on the low side, sentence if on the high side) |

**Routes**
| Code | Meaning |
|---|---|
| `@app.route("/", methods=["GET"])` | "When someone opens `/` with GET, run the function below" (the only decorator) |
| `def home(): return "Loan Risk API is running"` | A health check |
| `@app.route("/predict", methods=["POST"])` | Same, but for `/predict` and only POST |
| `def predict():` | Function that runs on each prediction request |

**Inside `predict()`**
| Step | Code | Meaning |
|---|---|---|
| 1 | `if model is None: return jsonify({...}), 500` | Model missing → send an error message; `500` = server problem |
| 2 | `data = request.get_json()` | Read the JSON from React into dictionary `data` |
| 2 | `if data is None: ... 400` | Nothing was sent → error; `400` = client mistake |
| 3 | `for name in feature_names: if name not in data or str(data[name]).strip() == ""` | Loop over each expected field; error if missing or empty |
| 4 | `if data["employment_status"] not in employment_map ...` | Reject unknown words |
| 4 | `float(data["age"])` ... inside `try:` | React sends everything as text; `float()` makes numbers. If text like "abc" → `ValueError` is caught by `except ValueError:` and we return an error |
| 4 | `if age < 18 or credit_score < 300 ...` | Simple range check |
| 4 | `employment_number = employment_map[...]` | Word → number |
| 5 | `applicant = [[...9 numbers...]]` | Model expects a table (list of rows). We have one row, hence double brackets |
| 6 | `risk = model.predict(applicant)[0]` | Predict. Result is a list with one item, `[0]` takes it |
| 6 | `model.predict_proba(applicant)[0]` | Probability of each class; `max(...)` is the confidence |
| 7 | `tree = model.tree_` | The tree's internal structure (arrays describing every node) |
| 7 | `node_ids = model.decision_path(applicant).indices` | The list of node numbers this applicant visited from root to leaf |
| 7 | `for node_id in node_ids:` | Visit each node |
| 7 | `if tree.children_left[node_id] == -1: continue` | A leaf has no children (`-1`); a leaf is the answer, not a question, so skip it |
| 7 | `feature_index = tree.feature[node_id]` | Which column this node asked about |
| 7 | `threshold = tree.threshold[node_id]` | The cut-off number (e.g. 595) |
| 7 | `value = applicant[0][feature_index]` | The applicant's own value for that column |
| 7 | `if feature_name == "employment_status": ...` | For word columns, show the original word, not 0/1/2 |
| 7 | `elif value <= threshold:` | Tree went left → add "X <= threshold" and the "low side" sentence |
| 7 | `else:` | Tree went right → add "X > threshold" and the "high side" sentence |
| 7 | `decision_path.append("Prediction = " + risk)` | Final step of the path |
| 8 | `return jsonify({...})` | Send the JSON back |
| end | `app.run(debug=True)` | Start the server. `debug=True` shows errors and auto-restarts on code change |

**Why the explanation is genuine:** every item in `decision_path` and `reasons` comes from a node the model *actually used* for this applicant. The only hand-written part is the wording of the sentences.

## 3.3 `train_model.py` line-by-line
| Code | Meaning |
|---|---|
| `import pandas as pd` | pandas, nicknamed `pd` |
| `train_test_split`, `DecisionTreeClassifier`, `export_text`, `SVC`, `StandardScaler`, `accuracy_score`, `joblib` | Tools: split data, tree, print tree, SVM, scale numbers, measure accuracy, save file |
| `data = pd.read_csv("loan_data.csv")` | Load CSV into a table |
| `data["employment_status"].map(employment_map)` | Replace each word with its number |
| `X = data[feature_names].values` | Input columns as plain numbers |
| `y = data["risk"]` | The answers |
| `train_test_split(X, y, test_size=0.25, random_state=42)` | Returns 4 parts: X_train, X_test, y_train, y_test |
| `DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=2, random_state=42)` | Create an (empty) tree. `entropy` = ID3-style; `max_depth=5` = at most 5 questions in a row; `min_samples_leaf=2` = no leaf with fewer than 2 rows (both stop the tree from memorising → less overfitting) |
| `tree_model.fit(X_train, y_train)` | **Training**: builds the tree using information gain |
| `tree_model.predict(X_test)` | Predict the hidden test rows |
| `accuracy_score(y_test, predictions)` | Compare with real answers |
| `StandardScaler()` ... `SVC(kernel="linear")` | Optional SVM: scale numbers, train, test |
| `export_text(tree_model, feature_names=...)` | Print the tree as readable rules |
| `joblib.dump(tree_model, "model.pkl")` | Save the trained tree so Flask can reuse it without retraining |

---

# PART 4 — React

## 4.1 Concepts
- **Component** = a JavaScript function that returns HTML-like code (JSX). Our whole app is one component, `App`.
- **State** (`useState`) = memory of a component. When state changes, React redraws the screen automatically. We have 3: `form` (typed values), `result` (Flask's answer), `error` (message).
- **Controlled input** = each input shows `value={form.age}` and calls `onChange={handleChange}` when typed in, so React always knows what's typed.
- **Button click** = `onClick={handlePredict}` runs our function.
- **fetch** = built-in browser function to make an HTTP request (like `$.ajax` in jQuery).
- **POST request** = `method: "POST"` with `body: JSON.stringify(form)`.
- **Receiving JSON** = `await response.json()`.
- **Displaying** = `{result.risk}` inside JSX; lists with `.map()`.

## 4.2 `App.jsx` line-by-line

| Code | Meaning |
|---|---|
| `import { useState } from "react"` | Get the state tool |
| `function App() {` | Define the component |
| `const [form, setForm] = useState({...})` | `form` = current value, `setForm` = the only way to change it. Starts with an example applicant (all values are text, like inputs give) |
| `const [result, setResult] = useState(null)` | `null` = no answer yet |
| `const [error, setError] = useState("")` | Empty text = no error |
| `function handleChange(event)` | Runs on every keystroke. `event` describes what happened |
| `setForm({ ...form, [event.target.name]: event.target.value })` | `...form` copies all old fields; `[event.target.name]: ...` replaces the one that changed. Each input has a `name` like `"age"`, so one function serves all 9 inputs |
| `async function handlePredict()` | `async` allows using `await` inside, i.e. "wait for the server's reply" |
| `setError(""); setResult(null)` | Clear the old messages |
| `try { ... } catch (err) { ... }` | Run the code; if it fails (e.g. Flask is off) jump to `catch` |
| `await fetch("http://127.0.0.1:5000/predict", {...})` | Send the request and wait |
| `method: "POST"` | Use POST |
| `headers: { "Content-Type": "application/json" }` | Tell Flask the body is JSON |
| `body: JSON.stringify(form)` | Turn the form object into JSON text |
| `const data = await response.json()` | Read the reply |
| `if (response.ok) setResult(data) else setError(data.error)` | `ok` is true for status 200. Otherwise show Flask's error message |
| `catch → setError("Cannot reach the backend...")` | Network failure message |
| `return ( <div className="page"> ...` | What to draw. In JSX we write `className` instead of `class` |
| `<label>Age <input ... /></label>` | A text label with its input; `type="number"` accepts only numbers |
| `<select ...><option>...</option></select>` | Dropdown for the two text fields |
| `<button onClick={handlePredict}>` | Click → run prediction |
| `{error && <div className="error">{error}</div>}` | "If error is not empty, show the box" |
| `{result && ( ... )}` | Show results only after Flask replied |
| `className={"risk " + result.risk.split(" ")[0].toLowerCase()}` | "High Risk" → "High" → "high", giving class `risk high` so CSS can colour it |
| `Math.round(result.confidence * 100)` | 0.96 → 96 |
| `result.reasons.map((reason, index) => (<li key={index}>{reason}</li>))` | Turn each reason into a bullet. `key` is a unique tag React needs for list items |
| `<ol>` for the path | Ordered (numbered) list since the path is a sequence |
| `export default App` | Lets `main.jsx` import it |

`main.jsx` finds `<div id="root">` in `index.html` and draws `<App />` inside it. `index.html` is the single page; `package.json` lists the libraries and the `npm run dev` command.

---

# PART 5 — CSS (`App.css`)
| Rule | Why |
|---|---|
| `body { margin: 0; background; font-family }` | Remove browser's default margin, set page colour and font |
| `.page { max-width: 760px; margin: 0 auto; }` | Content never wider than 760px; `auto` side margins centre it |
| `.card { background: white; border-radius; padding; border }` | White rounded box for the form and the result |
| `.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px }` | Two equal columns of fields |
| `label { display: flex; flex-direction: column }` | Label text above its input |
| `input, select { padding; border }` | Comfortable-looking inputs |
| `button:hover` | Colour change when the mouse is over it |
| `.risk.low / .medium / .high` | Green / amber / red, chosen by the extra class set in JSX |
| `.error` | Red box for errors |
| `@media (max-width: 560px)` | On phones, switch to one column |

---

# PART 6 — Complete execution flow (real example)

User enters: Age 25, Income 60000, Salaried, Loan Amount 300000, Term 36, Credit Score 750, Existing Loans 0, DTI 20, Previous = Repaid, then clicks **Predict Loan Risk**.

1. **React** `handlePredict` runs and `fetch` sends:
```json
{"age":"25","monthly_income":"60000","employment_status":"Salaried","loan_amount":"300000",
 "loan_term":"36","credit_score":"750","existing_loans":"0","debt_to_income":"20",
 "previous_repayment":"Repaid"}
```
2. **Flask** `/predict` reads it, checks nothing is missing, converts to numbers (Salaried → 2, Repaid → 2):
```
[[25, 60000, 2, 300000, 36, 750, 0, 20, 2]]
```
3. **Model** walks the tree:
   - `credit_score 750 > 595` → right
   - `previous_repayment 2 > 1.5` (Repaid) → right
   - `existing_loans 0 <= 2.5` → left
   - `monthly_income 60000 > 40000` → right
   - `debt_to_income 20 <= 35` → left → **leaf: Low Risk**, 96% of training rows in this leaf were Low Risk.
4. **Flask** builds and returns (real output):
```json
{
  "risk": "Low Risk",
  "confidence": 0.96,
  "reasons": [
    "Credit score is high (750 > 595)",
    "Previous loan repayment: Repaid",
    "Number of existing loans is low (0 <= 2.5)",
    "Monthly income is sufficient (60000 > 40000)",
    "Debt-to-income ratio is low (20 <= 35)"
  ],
  "decision_path": [
    "Credit Score > 595",
    "Previous Loan Repayment = Repaid",
    "Existing Loans <= 2.5",
    "Monthly Income (Rs) > 40000",
    "Debt-to-Income Ratio (%) <= 35",
    "Prediction = Low Risk"
  ]
}
```
5. **React** saves it with `setResult(data)`, redraws, and shows a green "Low Risk", "96%", the reasons list and the numbered path.

Try a second case: Credit Score 560 → path is just `Credit Score <= 595` → High Risk (100%). One question was enough, which shows how explanations adapt to each applicant.

---

# Error handling cheat-sheet
| Problem | What you see | Fix |
|---|---|---|
| Missing input | Red box: "Please fill in: Monthly Income (Rs)" | Fill the field |
| Invalid input (letters, credit score 1200, age 10) | Red box with message from Flask | Enter valid numbers |
| Flask not running | "Cannot reach the backend. Is Flask running on port 5000?" | Start `python app.py` in the backend terminal with `(venv)` active |
| `model.pkl` missing | "model.pkl not found. Run train_model.py first." | In `backend`, run `python train_model.py`, then restart Flask |
| `model.pkl` not found even after training | Run `python app.py` from a different folder | `cd backend` first; the file is loaded relative to where you run |
| CORS error in browser console (F12) | "blocked by CORS policy" | `pip install flask-cors`; check `CORS(app)` is in `app.py`; restart Flask |
| `ModuleNotFoundError` | Python can't find a library | Activate the venv (`venv\Scripts\Activate.ps1`) and `pip install` it |
| `npm` not recognised | PowerShell error | Install Node.js, then reopen VS Code |
| Port 5000 in use | Flask won't start | Close the other program or change to `app.run(port=5001)` AND the URL in `App.jsx` |

---

# PART 7 — Viva questions and answers

**1. Why did you choose ID3?**
Because the project must *explain* its predictions. A decision tree is a set of yes/no questions that a human can read, so every prediction has a visible reason.

**2. What is ID3?**
An algorithm that builds a decision tree top-down. At each step it picks the attribute with the highest information gain, splits the data on it, and repeats on each branch.

**3. What is entropy?**
A measure of impurity or disorder. Zero means all rows in a group have the same class; higher means the classes are mixed. Formula: −Σ p log2 p.

**4. What is information gain?**
Entropy before a split minus the weighted entropy after it. It tells how much a question reduces uncertainty. The question with the largest gain is chosen. In my data the root question `credit_score <= 595` had gain ≈ 0.50.

**5. Why is a decision tree explainable?**
Each prediction is a path of simple conditions from root to leaf. We can print that path and turn it into sentences, and it is exactly what the model did.

**6. What is SVM?**
Support Vector Machine: it finds the boundary (hyperplane) that separates classes with the maximum margin.

**7. Why didn't you use SVM as the primary model?**
It is harder to explain: the decision depends on a weighted combination of all features, so we can't give a simple "because X" path. Also it needs feature scaling. I used it only for comparison (tree 92% vs SVM 84%).

**8. What is overfitting?**
When the model memorises the training data, including noise, and does badly on new data. A fully grown tree overfits easily. I reduced it with `max_depth=5` and `min_samples_leaf=2`.

**9. What is train-test split?**
Dividing the data into one part for learning (75%) and one part kept hidden for checking (25%), so accuracy reflects performance on unseen data.

**10. What is accuracy?**
Correct predictions divided by total predictions on the test set. Mine is 92% (23 out of 25).

**11. What is Flask?**
A lightweight Python web framework used to create a server with API routes.

**12. Why do we need Flask?**
To make the Python model reachable from the browser. React sends data to Flask, Flask runs the model and replies.

**13. Why can't React directly run the Python ML model?**
React is JavaScript running in the browser; the model is a Python/scikit-learn object. The browser can't run Python, so a Python server must sit in between.

**14. What is an API?**
A defined way for two programs to communicate. Here: React sends a request to the URL `/predict` and receives data back.

**15. What is JSON?**
A text format for exchanging data using key–value pairs, readable by both JavaScript and Python.

**16. What is POST?**
An HTTP method used to send data to the server in the request body. I use it because I'm sending a full form and expecting a computed result.

**17. What is CORS?**
A browser security rule that blocks a web page from reading responses from a different origin (different port counts) unless the server allows it. `flask-cors` makes Flask allow React.

**18. How does React communicate with Flask?**
With `fetch`: a POST request to `http://127.0.0.1:5000/predict` with the form as JSON; Flask replies with JSON, which React stores in state and displays.

**19. How is the loan risk calculated?**
The 9 inputs become numbers, then follow the learned tree from the root, answering each question, until a leaf. The leaf's majority class is the risk.

**20. How does the explanation come from the model?**
Flask asks the tree for the nodes this applicant passed (`decision_path`), reads each node's feature and threshold from `tree_`, compares with the applicant's value, and converts it to text. Only the sentence wording is hand-written.

**21. What happens when the user clicks Predict?**
`handlePredict` runs → `fetch` POST → Flask validates, converts and predicts → returns JSON → React sets `result` → screen updates.

**22. What happens if invalid input is entered?**
Flask checks for missing values, non-numbers and out-of-range values and returns a 400 error with a message; React shows it in a red box.

**23. How could this project be improved?**
Use real, larger data; cross-validation; compare more models; feature importance; scale or one-hot encode categories properly; better input validation in React; deploy online; show a visual tree.

**24. What are the limitations?**
Synthetic and small dataset (100 rows), so accuracy isn't realistic for a real bank; the tree can be biased by the data; only a single train/test split; the explanation shows the model's reasoning, not whether the reasoning is fair; not suitable for real lending decisions.

**25. What is the difference between ID3 and SVM?**
ID3 builds a tree of rules using entropy; it's easy to read and needs no scaling. SVM finds a maximum-margin boundary; it can handle complex patterns but is hard to explain and needs scaled data.

**26. What is the difference between ID3 and CART?**
ID3 uses information gain and makes multi-way splits on categories. CART always makes binary splits and uses Gini by default. scikit-learn implements CART; with `criterion="entropy"` it uses ID3's entropy idea.

**27. Is scikit-learn's DecisionTreeClassifier really ID3?**
No, strictly it's an optimised CART. Setting `criterion="entropy"` makes the split selection follow ID3's information gain, so I describe it as an ID3-style tree.

**28. Why do you convert Salaried/Repaid etc. to numbers?**
Models can only do arithmetic and comparisons on numbers. A dictionary maps each word to a number, and the same mapping is used in training and prediction.

**29. What does `random_state=42` do?**
It fixes the random shuffle so the split and tree are reproducible. The number 42 itself has no special meaning.

**30. What does confidence mean here?**
The proportion of training rows in the final leaf that belong to the predicted class. 0.96 means 96% of similar training applicants had that label. It is not a guarantee.

**31. Why is the dataset synthetic?**
Real loan data is private. A small synthetic dataset lets me see every row and understand the model; the focus is the system and the explainability.

**32. What does `model.pkl` contain?**
The trained decision tree (its questions and thresholds), saved with joblib so Flask doesn't retrain on every request.

---

# Possible future improvements
1. Real dataset (e.g. German Credit) and cross-validation instead of one split.
2. Random Forest or gradient boosting plus tools like SHAP/LIME for explaining them.
3. One-hot encoding for categories; more features (EMI-to-income ratio).
4. Input validation in React before sending.
5. Draw the tree visually (graphviz / `plot_tree`).
6. Save prediction history in a database; user login; deployment.
7. Fairness checks (is the model biased against certain groups?).

# Project limitations
- Synthetic, rule-generated data: the model largely re-learns the rules used to create it.
- Only 100 rows: results change a lot with a different split (25 test rows, so each wrong prediction costs 4%).
- Single train/test split; no cross-validation.
- Tree is CART with entropy, not a textbook multi-way ID3.
- Ordinal encoding assumes an order for categories.
- Explanations show the model's logic, not true cause-and-effect or fairness.
- Not for real lending decisions.
