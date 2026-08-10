# Telco Customer Churn Prediction

## 🚀 Live Demo

**Live Streamlit App:** `https://customer-churn-0809.streamlit.app/`

---

## 📌 Project Overview

This project predicts whether a telecom customer is likely to **churn** or **stay** based on customer demographics, services, contract details, and billing information.

The project covers the complete machine learning workflow:

**Data Collection → Data Cleaning → EDA → Feature Preparation → Encoding → Scaling → Train/Test/Unseen Split → Cross Validation → Model Comparison → Final Model Selection → Pickle Model → Streamlit Deployment**

The final application allows users to enter customer information and receive:

- Churn / Stay prediction
- Churn probability
- A simple interactive Streamlit interface

---

## 🎯 Problem Statement

Customer churn is an important business problem for telecom companies.

If a company can identify customers who are likely to leave, it can take preventive actions such as:

- Offering discounts
- Providing better support
- Recommending suitable plans
- Improving customer experience
- Creating customer retention campaigns

The objective of this project is to build a classification model that predicts whether a customer will churn.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

### Dataset Size

- **Rows:** 7,043
- **Columns:** 21
- **Target:** `Churn`

The original dataset contains a `customerID` column. This identifier is not useful for prediction, so it is excluded from the model features.

### Target Variable

| Value | Meaning |
|---|---|
| `Yes` | Customer will churn |
| `No` | Customer will stay |

For machine learning, the target was label encoded.

---

## 🧾 Features Used

The model uses the following 19 input features:

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `PhoneService` | Whether phone service is subscribed |
| `MultipleLines` | Multiple phone lines subscription |
| `InternetService` | Type of internet service |
| `OnlineSecurity` | Online security subscription |
| `OnlineBackup` | Online backup subscription |
| `DeviceProtection` | Device protection subscription |
| `TechSupport` | Technical support subscription |
| `StreamingTV` | Streaming TV subscription |
| `StreamingMovies` | Streaming movies subscription |
| `Contract` | Contract type |
| `PaperlessBilling` | Paperless billing status |
| `PaymentMethod` | Payment method |
| `MonthlyCharges` | Monthly customer charge |
| `TotalCharges` | Total amount charged |

`customerID` was removed because it is an identifier rather than a meaningful predictive feature.

---

# 🔄 Machine Learning Workflow

## 1. Data Loading

The Telco Customer Churn CSV dataset was loaded using Pandas.

## 2. Data Cleaning

The dataset was inspected for:

- Missing values
- Incorrect data types
- Duplicate records
- Unnecessary columns
- Numerical and categorical features

## 3. Feature Separation

The target was separated from the input features:

```python
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]
```

The target was encoded using `LabelEncoder`.

## 4. Categorical and Numerical Features

Categorical and numerical columns were identified automatically:

```python
cat_cols = X.select_dtypes(include="object").columns.tolist()

num_cols = X.select_dtypes(
    include=["int", "float"]
).columns.tolist()
```

## 5. Feature Scaling

Numerical features were standardized using:

```text
StandardScaler
```

## 6. Categorical Encoding

Categorical features were converted using:

```text
OneHotEncoder
```

with:

- `handle_unknown="ignore"`
- `drop="first"`
- `sparse_output=False`

This ensures that the Streamlit application can handle categorical values consistently.

---

# ✂️ Data Splitting

The dataset was divided into:

- **90% remaining data**
- **10% unseen data**

The remaining 90% was further divided into:

- **80% training**
- **20% testing**

This gives approximately:

```text
72% Training
18% Testing
10% Unseen
```

The unseen dataset was kept separate for final evaluation.

---

# 🤖 Models Compared

Eight classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. XGBoost
7. LightGBM
8. CatBoost

Each model was evaluated using **5-fold Cross Validation**.

---

# 📈 Model Comparison

The final comparison produced the following results:

| Model | CV Mean Accuracy | CV Std | Test Accuracy | Unseen Accuracy |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **80.32%** | 0.0076 | 79.89% | **81.70%** |
| Gradient Boosting | 79.53% | 0.0052 | 79.97% | 81.56% |
| LightGBM | 78.88% | 0.0078 | 79.57% | 80.99% |
| XGBoost | 80.22% | 0.0089 | **80.84%** | 80.85% |
| Random Forest | 79.72% | 0.0104 | 80.13% | 80.00% |
| CatBoost | 78.99% | 0.0101 | 79.50% | 78.87% |
| AdaBoost | 79.29% | 0.0079 | 80.05% | 78.44% |
| Decision Tree | 78.15% | 0.0145 | 79.10% | 78.44% |

---

# 🏆 Final Model

## Logistic Regression

**Logistic Regression was selected as the final deployed model.**

Reasons:

- Highest CV Mean Accuracy: **80.32%**
- Highest Unseen Accuracy: **81.70%**
- Stable performance between validation and unseen data
- Simple and efficient classification algorithm
- Easy to interpret
- Suitable for a lightweight production application

Although XGBoost achieved the highest test accuracy (**80.84%**), Logistic Regression achieved the best unseen accuracy (**81.70%**) and the highest cross-validation mean accuracy.

Therefore, Logistic Regression was selected for deployment.

---

# 🧪 Final Evaluation

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score
- 5-fold Cross Validation
- Unseen Dataset Accuracy

The unseen dataset was kept separate from model training and used as a final performance check.

---

# 💾 Model Saving

The final deployment uses two pickle files:

```text
churn_model.pkl
preprocessor.pkl
```

### `churn_model.pkl`

Contains the trained Logistic Regression model.

### `preprocessor.pkl`

Contains the fitted preprocessing object responsible for:

- Numerical scaling
- Categorical one-hot encoding
- Feature transformation

The Streamlit application follows this process:

```text
User Input
    ↓
Pandas DataFrame
    ↓
preprocessor.pkl
    ↓
Encoded + Scaled Features
    ↓
churn_model.pkl
    ↓
Prediction
    ↓
Churn / Stay
```

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

- Gender
- Senior Citizen status
- Partner status
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application returns:

### Prediction

```text
Customer is likely to Churn
```

or

```text
Customer is likely to Stay
```

### Churn Probability

The application also displays the probability that the customer will churn.

---

# 📁 Project Structure

```text
customer-churn/
│
├── README.md
├── app.py
├── churn_model.pkl
├── preprocessor.pkl
├── customer churn.ipynb
├── Telco-Customer-Churn.csv
└── requirements.txt
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `churn_model.pkl` | Trained Logistic Regression model |
| `preprocessor.pkl` | Fitted preprocessing object |
| `customer churn.ipynb` | Complete ML workflow and experimentation |
| `Telco-Customer-Churn.csv` | Dataset |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

# ⚙️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Model Deployment

- Streamlit
- Pickle

### Development Environment

- Jupyter Notebook
- VS Code

### Version Control

- Git
- GitHub

---

# 📦 Installation

Clone the repository and move into the project directory.

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# ☁️ Streamlit Cloud Deployment

This application can be deployed using Streamlit Community Cloud.

Required files for deployment:

```text
app.py
churn_model.pkl
preprocessor.pkl
requirements.txt
```

After connecting the GitHub repository:

1. Select the repository.
2. Select the `customer-churn` folder if applicable.
3. Select `app.py` as the main file.
4. Deploy the application.
5. Copy the generated live application URL.
6. Replace the Live Demo placeholder at the top of this README.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

- Classification
- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Categorical Encoding
- One-Hot Encoding
- Label Encoding
- Feature Scaling
- Train/Test Split
- Unseen Data Evaluation
- Cross Validation
- Model Comparison
- Accuracy Evaluation
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score
- Model Selection
- Pickle Model Serialization
- Streamlit Deployment

---

# 📌 Business Interpretation

The model can help a telecom company identify customers who have a higher probability of leaving.

Potential business actions include:

- Targeted retention offers
- Personalized plans
- Customer support follow-ups
- Contract incentives
- Service quality improvements
- Loyalty programs

The model should be treated as a decision-support system rather than the only factor used to make business decisions.



# 👨‍💻 Author

**Bansi Sojitra**

GitHub: **BansiSojitra21**

---
