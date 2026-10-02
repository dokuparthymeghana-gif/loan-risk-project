// main.jsx - the starting point of the React app
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./App.css";

// Find <div id="root"> in index.html and draw the App component inside it
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
