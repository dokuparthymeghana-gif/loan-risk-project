// App.jsx - the only component in this project
import { useState } from "react";
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
function App() {
  // ---------- STATE ----------
  // form = what the user has typed (starts with an example applicant)
  const [form, setForm] = useState({
    age: "25",
    monthly_income: "60000",
    employment_status: "Salaried",
    loan_amount: "300000",
    loan_term: "36",
    credit_score: "750",
    existing_loans: "0",
    debt_to_income: "20",
    previous_repayment: "Repaid",
  });
  const [result, setResult] = useState(null); // answer from Flask (null = nothing yet)
  const [error, setError] = useState("");     // error message to show

  // Runs every time the user types in any input box
  function handleChange(event) {
    // event.target.name  = which field (e.g. "age")
    // event.target.value = what was typed
    // Copy the old form, then replace only the changed field
    setForm({ ...form, [event.target.name]: event.target.value });
  }

  // Runs when the user clicks "Predict Loan Risk"
  async function handlePredict() {
    setError("");     // clear old error
    setResult(null);  // clear old result
    try {
      // Send the form to Flask using a POST request
     const response = await fetch(API_URL + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form), // convert the form object to JSON text
      });
      const data = await response.json(); // read Flask's JSON reply

      if (response.ok) {
        setResult(data);        // success: save the prediction
      } else {
        setError(data.error);   // Flask sent an error message
      }
    } catch (err) {
      // Happens when Flask is not running / cannot be reached
      setError("Cannot reach the backend. Is Flask running on port 5000?");
    }
  }

  // ---------- WHAT IS SHOWN ON THE SCREEN ----------
  return (
    <div className="page">
      <h1>Explainable Loan Risk Prediction</h1>
      <p className="subtitle">Enter applicant details to get a risk prediction and the reasons behind it.</p>

      <div className="card">
        <h2>Applicant Form</h2>
        <div className="form-grid">
          <label>Age
            <input type="number" name="age" value={form.age} onChange={handleChange} />
          </label>
          <label>Monthly Income (Rs)
            <input type="number" name="monthly_income" value={form.monthly_income} onChange={handleChange} />
          </label>
          <label>Employment Status
            <select name="employment_status" value={form.employment_status} onChange={handleChange}>
              <option>Salaried</option>
              <option>Self-Employed</option>
              <option>Unemployed</option>
            </select>
          </label>
          <label>Loan Amount (Rs)
            <input type="number" name="loan_amount" value={form.loan_amount} onChange={handleChange} />
          </label>
          <label>Loan Term (months)
            <input type="number" name="loan_term" value={form.loan_term} onChange={handleChange} />
          </label>
          <label>Credit Score (300-900)
            <input type="number" name="credit_score" value={form.credit_score} onChange={handleChange} />
          </label>
          <label>Existing Loans
            <input type="number" name="existing_loans" value={form.existing_loans} onChange={handleChange} />
          </label>
          <label>Debt-to-Income Ratio (%)
            <input type="number" name="debt_to_income" value={form.debt_to_income} onChange={handleChange} />
          </label>
          <label>Previous Loan Repayment
            <select name="previous_repayment" value={form.previous_repayment} onChange={handleChange}>
              <option>Repaid</option>
              <option>No History</option>
              <option>Defaulted</option>
            </select>
          </label>
        </div>
        <button onClick={handlePredict}>Predict Loan Risk</button>
      </div>

      {/* Show the error box only if error is not empty */}
      {error && <div className="error">{error}</div>}

      {/* Show the result card only after Flask has replied */}
      {result && (
        <div className="card">
          <h2>Prediction</h2>
          <p className={"risk " + result.risk.split(" ")[0].toLowerCase()}>{result.risk}</p>

          <h2>Confidence</h2>
          <p className="confidence">{Math.round(result.confidence * 100)}%</p>

          <h2>Why?</h2>
          <ul>
            {result.reasons.map((reason, index) => (
              <li key={index}>{reason}</li>
            ))}
          </ul>

          <h2>Decision Path</h2>
          <ol className="path">
            {result.decision_path.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default App;
