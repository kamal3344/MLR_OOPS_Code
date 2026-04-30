# 🚀 Multiple Linear Regression — OOP Implementation with Flask Deployment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A complete, production-style implementation of **Multiple Linear Regression** using **Object-Oriented Programming (OOP)** in Python, trained on the 50 Startups dataset, and served through a **Flask web application**.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [What is Multiple Linear Regression?](#-what-is-multiple-linear-regression)
3. [Mathematical Foundation](#-mathematical-foundation)
4. [Dataset Description](#-dataset-description)
5. [Project Structure](#-project-structure)
6. [OOP Design — The MLR Class](#-oop-design--the-mlr-class)
7. [Code Walkthrough — main.py](#-code-walkthrough--mainpy)
8. [Exception Handling](#-exception-handling)
9. [Model Persistence with Pickle](#-model-persistence-with-pickle)
10. [Flask Web Application — app.py](#-flask-web-application--apppy)
11. [Frontend — templates/index.html](#-frontend--templatesindexhtml)
12. [Evaluation Metrics](#-evaluation-metrics)
13. [Installation & Usage](#-installation--usage)
14. [How to Run](#-how-to-run)
15. [Sample Prediction](#-sample-prediction)
16. [Technologies Used](#-technologies-used)
17. [Contact & Support](#-contact--support)

---

## 📌 Project Overview

This project demonstrates how to build a **Multiple Linear Regression (MLR)** model in a clean, maintainable, and production-ready way using:

- **OOP (Object-Oriented Programming)** principles to encapsulate the entire ML pipeline inside a single `MLR` class.
- **Exception handling** with detailed error diagnostics (line number, error type, and message).
- **Model serialization** using Python's `pickle` module.
- **Flask** to serve predictions through a simple, interactive web interface.

The model predicts the **profit of a startup company** based on its R&D Spend, Administration costs, Marketing Spend, and geographic State.

---

## 🧠 What is Multiple Linear Regression?

**Multiple Linear Regression (MLR)** is a supervised machine learning algorithm used to model the relationship between **one dependent variable (target)** and **two or more independent variables (features)**.

Unlike Simple Linear Regression (which uses only one predictor), MLR captures the combined influence of multiple features on the outcome.

### Real-World Use Cases
- Predicting house prices based on area, rooms, location, and age.
- Estimating startup profits from departmental spending.
- Forecasting sales from advertising budget across multiple channels.
- Predicting employee salaries from experience, education, and department.

### Key Assumptions of MLR
1. **Linearity** — The relationship between predictors and the target is linear.
2. **Independence** — Observations are independent of each other.
3. **Homoscedasticity** — Constant variance of residuals.
4. **Normality** — Residuals are approximately normally distributed.
5. **No multicollinearity** — Independent variables should not be highly correlated with each other.

---

## 📐 Mathematical Foundation

The MLR equation is an extension of simple linear regression:

```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ + ε
```

Where:
| Symbol | Meaning |
|--------|---------|
| `y` | Dependent variable (Profit) |
| `β₀` | Intercept (bias term) |
| `β₁, β₂, ..., βₙ` | Coefficients for each feature |
| `x₁, x₂, ..., xₙ` | Independent variables (R&D, Admin, Marketing, State) |
| `ε` | Error term (residual) |

### For this project specifically:
```
Profit = β₀ + β₁(R&D Spend) + β₂(Administration) + β₃(Marketing Spend) + β₄(State)
```

The algorithm finds the best-fit hyperplane by **minimizing the Sum of Squared Residuals (SSR)** using the **Ordinary Least Squares (OLS)** method:

```
Minimize: SSR = Σ(yᵢ - ŷᵢ)²
```

The closed-form solution (Normal Equation):
```
β = (XᵀX)⁻¹ Xᵀy
```

---

## 📊 Dataset Description

**File:** `50_Startups.csv`

This dataset contains data on 50 startup companies and their financial metrics:

| Column | Type | Description |
|--------|------|-------------|
| `R&D Spend` | Float | Amount spent on Research & Development |
| `Administration` | Float | Administration expenses |
| `Marketing Spend` | Float | Money spent on marketing |
| `State` | String → Integer | Geographic location (encoded: New York=0, California=1, Florida=2) |
| `Profit` | Float | **Target variable** — Company profit |

**Dataset Split:**
- Training set: **80%** (40 samples)
- Testing set: **20%** (10 samples)
- `random_state=42` ensures reproducibility

### Label Encoding for State
The `State` column is a categorical variable encoded into integers:
```python
{'New York': 0, 'California': 1, 'Florida': 2}
```

---

## 🗂️ Project Structure

```
MLR_OOPS_Code/
│
├── main.py                  # Core ML pipeline using OOP (MLR class)
├── app.py                   # Flask backend for web prediction
├── 50_Startups.csv          # Dataset
├── Model.pkl                # Trained & serialized model
├── templates/
│   └── index.html           # Frontend HTML form for user input
└── README.md                # Project documentation
```

---

## 🏗️ OOP Design — The MLR Class

The entire ML pipeline is encapsulated inside a single Python class `MLR`, following **Object-Oriented Programming** principles:

### Class Diagram

```
┌─────────────────────────────────────────────┐
│                   MLR                        │
├─────────────────────────────────────────────┤
│ Attributes:                                  │
│  + path : str                                │
│  + df : DataFrame                            │
│  + X : DataFrame (features)                  │
│  + y : Series (target)                       │
│  + X_train, X_test : DataFrame               │
│  + y_train, y_test : Series                  │
│  + reg : LinearRegression                    │
│  + y_train_predictions : array               │
│  + y_test_predictions : array                │
├─────────────────────────────────────────────┤
│ Methods:                                     │
│  + __init__(path)   → loads & splits data    │
│  + training()       → fits the model         │
│  + testing()        → evaluates on test set  │
│  + check_own_data() → custom prediction      │
│  + saving_model()   → pickle save & reload   │
└─────────────────────────────────────────────┘
```

### OOP Principles Applied

| Principle | How It's Applied |
|-----------|-----------------|
| **Encapsulation** | All data attributes (`self.df`, `self.X`, `self.reg`, etc.) are encapsulated within the `MLR` class |
| **Abstraction** | Complex operations (data loading, training, evaluation) are abstracted into clean method names |
| **Single Responsibility** | Each method has one clear responsibility (train, test, predict, save) |
| **Constructor Initialization** | `__init__` sets up all data pipelines when the object is created |

---

## 🔍 Code Walkthrough — main.py

### Imports & Setup

```python
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
```

- `numpy` & `pandas` — Data manipulation
- `sklearn` — Machine learning model and metrics
- `sys` — Used for detailed exception info via `sys.exc_info()`
- `warnings` — Suppresses non-critical scikit-learn deprecation warnings
- `pickle` — Model serialization/deserialization

---

### Method 1: `__init__(self, path)` — Data Loading & Splitting

```python
def __init__(self, path):
    try:
        self.path = path
        self.df = pd.read_csv(self.path)
        self.df['State'] = self.df['State'].map(
            {'New York': 0, 'California': 1, 'Florida': 2}
        ).astype(int)
        self.X = self.df.iloc[:, :-1]   # All columns except last
        self.y = self.df.iloc[:, -1]    # Last column (Profit)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(f"Training dataset size : {len(self.X_train)} : {len(self.y_train)}")
        print(f"Testing dataset size : {len(self.X_test)} : {len(self.y_test)}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

**What it does:**
- Reads the CSV file using `pd.read_csv()`
- Encodes the `State` column (categorical to numerical)
- Separates features (X) from target (y) using `iloc`
- Splits data 80/20 with `train_test_split`
- Prints dataset sizes for verification

---

### Method 2: `training(self)` — Model Training

```python
def training(self):
    try:
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)
        self.y_train_predictions = self.reg.predict(self.X_train)
        print(f"Train Accuracy : {r2_score(self.y_train, self.y_train_predictions)}")
        print(f"Train Loss : {root_mean_squared_error(self.y_train, self.y_train_predictions)}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

**What it does:**
- Instantiates scikit-learn's LinearRegression
- Fits the model on training data using .fit()
- Generates predictions on the training set
- Prints R2 score (accuracy) and RMSE (loss)

---

### Method 3: `testing(self)` — Model Evaluation

```python
def testing(self):
    try:
        self.y_test_predictions = self.reg.predict(self.X_test)
        print(f"Test Accuracy : {r2_score(self.y_test, self.y_test_predictions)}")
        print(f"Test Loss : {root_mean_squared_error(self.y_test, self.y_test_predictions)}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

**What it does:**
- Runs predictions on the held-out test set
- Evaluates model generalization using R2 and RMSE
- Detects overfitting by comparing train vs. test metrics

---

### Method 4: `check_own_data(self)` — Custom Prediction

```python
def check_own_data(self):
    try:
        rd    = 1200   # R&D Spend
        admin = 1800   # Administration
        ms    = 1900   # Marketing Spend
        s     = 1      # State (California)
        print(f"Test Point Predictions : {self.reg.predict([[rd, admin, ms, s]])[0]}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

**What it does:**
- Tests the trained model with a custom data point
- Useful for quickly verifying model output
- The [0] extracts the scalar from the NumPy array

---

### Method 5: `saving_model(self)` — Pickle Save & Verify

```python
def saving_model(self):
    try:
        with open("Model.pkl", "wb") as f:
            pickle.dump(self.reg, f)

        print("----------Load and check-------------")
        with open("Model.pkl", "rb") as t:
            model = pickle.load(t)
        rd, admin, ms, s = 1200, 1800, 1900, 1
        print(f"Loaded Model Predictions : {model.predict([[rd, admin, ms, s]])[0]}")
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

**What it does:**
- Serializes the trained LinearRegression object to Model.pkl using pickle.dump()
- Immediately reloads and verifies the saved model with pickle.load()
- Confirms the loaded model produces identical predictions

---

### Main Execution Block

```python
if __name__ == "__main__":
    try:
        path = "50_Startups.csv"
        obj = MLR(path)
        obj.training()
        obj.testing()
        obj.check_own_data()
        obj.saving_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

The `if __name__ == "__main__"` guard ensures this runs only when main.py is executed directly.

---

## 🛡️ Exception Handling

Every method in the MLR class uses a try-except block with detailed diagnostic output:

```python
except Exception as e:
    er_type, er_msg, er_line = sys.exc_info()
    print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")
```

### sys.exc_info() — Breakdown

sys.exc_info() returns a 3-tuple when called inside an except block:

| Variable | Value | Description |
|----------|-------|-------------|
| `er_type` | e.g., FileNotFoundError | The exception class |
| `er_msg` | e.g., No such file... | The error message/value |
| `er_line` | Traceback object | The traceback for stack inspection |
| `er_line.tb_lineno` | e.g., 18 | **Exact line number** where error occurred |

### Common Exceptions Handled

- **FileNotFoundError** — CSV file not found at the given path
- **KeyError** — Column name doesn't exist in the DataFrame
- **ValueError** — Data type mismatch during model training
- **AttributeError** — Calling testing() before training() (model not fitted)

---

## 💾 Model Persistence with Pickle

pickle is Python's built-in serialization module. It converts Python objects into a byte stream (and back), allowing models to be saved to disk and reloaded later.

```python
# Saving
with open("Model.pkl", "wb") as f:   # wb = write binary
    pickle.dump(self.reg, f)

# Loading
with open("Model.pkl", "rb") as t:   # rb = read binary
    model = pickle.load(t)
```

### Why Pickle?
- No need to retrain the model every time the web app starts
- app.py loads Model.pkl once at startup and reuses it for all predictions
- Fast and efficient for scikit-learn models

---

## 🌐 Flask Web Application — app.py

```python
from flask import Flask, render_template, request
import pickle, numpy as np

with open("Model.pkl", "rb") as f:
    m = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def check():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def fun3():
    a = [float(i) for i in request.form.values()]
    b = [np.array(a)]
    sol = m.predict(b)[0]
    return render_template("index.html", prediction_text=sol)

if __name__ == "__main__":
    app.run(debug=True)
```

### Route Details

| Route | Method | Handler | Description |
|-------|--------|---------|-------------|
| / | GET | check() | Renders the input form |
| /predict | POST | fun3() | Receives form data, runs prediction, returns result |

---

## 📈 Evaluation Metrics

### R² Score (Coefficient of Determination)

R² ranges from 0 to 1. A score of 1.0 means perfect prediction. Generally, R² above 0.9 is considered excellent for regression models.

### RMSE (Root Mean Squared Error)

RMSE is in the same unit as the target variable (USD profit). Lower RMSE = better accuracy. It penalizes large errors more heavily due to the squaring operation.

### Detecting Overfitting

| Scenario | Train R² | Test R² | Interpretation |
|----------|----------|---------|----------------|
| Good generalization | ~0.95 | ~0.93 | Model learned well |
| Overfitting | ~0.99 | ~0.70 | Model memorized training data |
| Underfitting | ~0.60 | ~0.58 | Model too simple |

---

## ⚙️ Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/kamal3344/MLR_OOPS_Code.git
cd MLR_OOPS_Code
```

### Install Dependencies

```bash
pip install numpy pandas scikit-learn flask
```

---

## ▶️ How to Run

### Step 1: Train the Model

```bash
python main.py
```

### Step 2: Launch the Flask App

```bash
python app.py
```

Open your browser: **http://localhost:5000/**

> **State Encoding:** Enter numeric values for State — 0 = New York, 1 = California, 2 = Florida

---

## 🎯 Sample Prediction

| Feature | Value |
|---------|-------|
| R&D Spend | 1200 |
| Administration | 1800 |
| Marketing Spend | 1900 |
| State | 1 (California) |

The model returns a predicted Profit value based on these inputs.

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core language |
| **NumPy** | 1.21+ | Numerical computation |
| **Pandas** | 1.3+ | Data manipulation |
| **scikit-learn** | 1.0+ | ML model & metrics |
| **Flask** | 2.0+ | Web framework |
| **Pickle** | Built-in | Model serialization |
| **Bootstrap** | 4.3.1 | Frontend styling |

---

## 📬 Contact & Support

**Sai Kamal Korlakunta**
*Data Scientist | NLP Engineer | Computer Vision | Blogger | Tech Speaker*

| Platform | Link |
|----------|------|
| 📧 Email | [saikamal9797@gmail.com](mailto:saikamal9797@gmail.com) |
| 💼 LinkedIn | [linkedin.com/in/sai-kamal-korlakunta-a81326163](https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163) |
| 🐙 GitHub | [github.com/kamal3344](https://github.com/kamal3344) |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ by [Sai Kamal](https://github.com/kamal3344) — Data Scientist & NLP Engineer*
