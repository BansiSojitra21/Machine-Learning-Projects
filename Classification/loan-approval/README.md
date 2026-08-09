# 🏦 Loan Approval Prediction

> **Machine Learning Classification Project** --- Predict whether a loan
> application is likely to be **Approved** or **Rejected** using
> applicant financial and personal information.

------------------------------------------------------------------------

## 🚀 Live Demo

### 👉 [Open Loan Approval Prediction App](https://loan-approval-prediction-03.streamlit.app/)

Try the deployed application directly in your browser. Enter applicant
details and get an instant loan approval prediction.

------------------------------------------------------------------------

## 📌 Project Overview

The **Loan Approval Prediction** project is a supervised Machine
Learning classification application designed to predict loan approval
based on an applicant's personal and financial information.

The project follows a complete end-to-end Machine Learning workflow:

``` text
Raw Dataset
    ↓
Data Cleaning & Exploration
    ↓
Feature Preparation
    ↓
Categorical Encoding
    ↓
Train / Test / Unseen Split
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Evaluation & Comparison
    ↓
Model Serialization
    ↓
Streamlit Application
    ↓
Cloud Deployment
```

------------------------------------------------------------------------

## 🎯 Objective

The main objective is to build a classification model that can predict:

  -----------------------------------------------------------------------
  Prediction                          Meaning
  ----------------------------------- -----------------------------------
  ✅ **Approved**                     The model predicts that the loan is
                                      likely to be approved

  ❌ **Rejected**                     The model predicts that the loan is
                                      unlikely to be approved
  -----------------------------------------------------------------------

The application also displays the model's **approval probability**.

> ⚠️ This project is for educational and demonstration purposes. The
> prediction should not be used as the sole basis for real financial or
> lending decisions.

------------------------------------------------------------------------

## 📊 Dataset

The project uses a loan approval dataset containing applicant
information and a target variable representing the loan decision.

### Features Used

  Feature              Description
  -------------------- ----------------------------------------
  `loan_id`            Unique loan application identifier
  `no_of_dependents`   Number of dependents
  `education`          Applicant's education status
  `self_employed`      Whether the applicant is self-employed
  `income_annum`       Applicant's annual income
  `loan_amount`        Requested loan amount
  `loan_term`          Loan repayment term
  `cibil_score`        Applicant's CIBIL / credit score
  `assets`             Applicant's total assets
  `loan_status`        Target: Approved / Rejected

------------------------------------------------------------------------

## 🔧 Data Preprocessing

### 1. Categorical Encoding

The categorical features were converted into numerical values before
model training.

#### Education

``` text
Graduate      → 1
Not Graduate  → 0
```

#### Self Employed

``` text
Yes → 1
No  → 0
```

#### Loan Status

``` text
Approved → 1
Rejected → 0
```

### 2. Feature Scaling

Numerical features were standardized using `StandardScaler`.

The fitted scaler is saved as:

``` text
scaler.pkl
```

The same scaler is loaded by the Streamlit application before making
predictions.

------------------------------------------------------------------------

## 🤖 Models Tested

Three classification algorithms were evaluated:

### Logistic Regression

-   Training Accuracy: \~92%
-   Unseen Accuracy: \~91%

### Decision Tree Classifier

-   Training Accuracy: \~98%
-   Unseen Accuracy: \~97%

### Random Forest Classifier

-   Training Accuracy: \~97%
-   Unseen Accuracy: \~96%

### 🏆 Selected Model

The **Decision Tree Classifier** was selected for deployment because it
achieved the best unseen-data performance among the tested models.

  Model                   Training Accuracy   Unseen Accuracy
  --------------------- ------------------- -----------------
  Logistic Regression                 \~92%             \~91%
  Random Forest                       \~97%             \~96%
  **Decision Tree**               **\~98%**         **\~97%**

> The exact values may vary slightly depending on the dataset version
> and preprocessing.

------------------------------------------------------------------------

## 📈 Model Evaluation

The models were evaluated using:

-   Accuracy Score
-   Precision
-   Recall
-   F1 Score
-   Classification Report
-   Confusion Matrix

Example confusion matrix for the selected Decision Tree model:

``` text
[[156   7]
 [  6 258]]
```

This indicates that the selected model performed strongly on the unseen
data used during evaluation.

------------------------------------------------------------------------

## 🌐 Streamlit Application

The trained model has been integrated into an interactive Streamlit web
application.

### User Inputs

The application accepts:

-   👥 Number of Dependents
-   🎓 Education
-   💼 Self Employed
-   💰 Annual Income
-   🏦 Loan Amount
-   📅 Loan Term
-   📊 CIBIL Score
-   🏠 Assets

### Prediction Output

After clicking **Predict Loan Status**, the application displays:

``` text
✅ Loan Approved
```

or

``` text
❌ Loan Rejected
```

along with the estimated approval probability.

------------------------------------------------------------------------

## 🖥️ How to Use the Web App

### Step 1 --- Open the application

Go to:

**[Loan Approval Prediction -- Live
App](https://loan-approval-prediction-03.streamlit.app/)**

### Step 2 --- Enter applicant information

Fill in:

1.  Number of dependents
2.  Education
3.  Self-employment status
4.  Annual income
5.  Loan amount
6.  Loan term
7.  CIBIL score
8.  Total assets

### Step 3 --- Predict

Click:

**🔍 Predict Loan Status**

### Step 4 --- View the result

The application will display the predicted loan status and approval
probability.

------------------------------------------------------------------------

## 📂 Files Overview

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  File                                                                                                                                                          Purpose
  ------------------------------------------------------------------------------------------------------------------------------------------------------------- -----------------------------------
  [`app.py`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/app.py)                                         Streamlit frontend and prediction
                                                                                                                                                                logic

  [`loan_approval.ipynb`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/loan_approval.ipynb)               Complete ML workflow including EDA,
                                                                                                                                                                preprocessing, training and
                                                                                                                                                                evaluation

  [`loan_approval_dataset.csv`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/loan_approval_dataset.csv)   Original dataset used for the
                                                                                                                                                                project

  [`loan_model.pkl`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/loan_model.pkl)                         Serialized trained Machine Learning
                                                                                                                                                                model

  [`scaler.pkl`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/scaler.pkl)                                 Serialized `StandardScaler` used
                                                                                                                                                                during preprocessing

  [`requirements.txt`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/requirements.txt)                     Python dependencies required to run
                                                                                                                                                                the application

  [`README.md`](https://github.com/BansiSojitra21/Machine-Learning-Projects/blob/main/Classification/loan-approval/README.md)                                   Project documentation
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🗂️ Project Structure

``` text
loan-approval/
│
├── 📄 app.py
├── 📓 loan_approval.ipynb
├── 📊 loan_approval_dataset.csv
├── 🤖 loan_model.pkl
├── ⚙️ scaler.pkl
├── 📦 requirements.txt
└── 📖 README.md
```

------------------------------------------------------------------------

## 🛠️ Technologies Used

### Programming Language

-   Python

### Data Science

-   Pandas
-   NumPy

### Data Visualization

-   Matplotlib
-   Seaborn

### Machine Learning

-   Scikit-learn

### Web Application

-   Streamlit

### Model Serialization

-   Pickle

### Deployment

-   Streamlit Community Cloud

------------------------------------------------------------------------

## ⚙️ Run the Project Locally

### 1. Clone the repository

``` bash
git clone https://github.com/BansiSojitra21/Machine-Learning-Projects.git
```

### 2. Navigate to the project

``` bash
cd Machine-Learning-Projects/Classification/loan-approval
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

``` bash
streamlit run app.py
```

The application will open in your browser, normally at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 🔐 How Model Prediction Works

The Streamlit application follows the same preprocessing logic used
during model training.

``` text
User Input
    ↓
Create DataFrame
    ↓
Convert categorical values to 0 / 1
    ↓
Load scaler.pkl
    ↓
Scale numerical features
    ↓
Load loan_model.pkl
    ↓
Generate prediction
    ↓
Display Loan Status
    ↓
Display Approval Probability
```

### Important

The deployed application does **not retrain the model**.

It loads the previously trained files:

``` text
loan_model.pkl
scaler.pkl
```

and uses them to make predictions.

------------------------------------------------------------------------

## 🧠 Machine Learning Workflow

The notebook contains the main Machine Learning development process:

### 1. Data Understanding

-   Dataset inspection
-   Shape and data types
-   Statistical summary
-   Missing-value analysis
-   Duplicate analysis

### 2. Exploratory Data Analysis

-   Distribution analysis
-   Categorical feature analysis
-   Numerical feature analysis
-   Relationship between features and target

### 3. Data Preparation

-   Feature selection
-   Categorical conversion
-   Numerical feature preparation
-   Train/Test/Unseen splitting

### 4. Model Training

The following models were trained and compared:

``` text
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
```

### 5. Evaluation

Models were evaluated using:

``` text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report
```

### 6. Deployment

The selected model and scaler were serialized using Pickle and
integrated into a Streamlit application.

------------------------------------------------------------------------

## 📌 Key Learning Outcomes

This project provided practical experience with:

-   Data cleaning
-   Exploratory Data Analysis
-   Feature preprocessing
-   Categorical encoding
-   Feature scaling
-   Train/Test splitting
-   Unseen-data validation
-   Classification algorithms
-   Model comparison
-   Confusion matrices
-   Classification metrics
-   Model serialization with Pickle
-   Streamlit application development
-   Cloud deployment

------------------------------------------------------------------------

## 🚀 Future Improvements

Potential improvements include:

-   Hyperparameter tuning
-   Cross-validation
-   Testing XGBoost and CatBoost
-   Better probability calibration
-   Explainable AI using SHAP
-   More robust preprocessing pipelines
-   Improved UI and visualizations
-   Model monitoring
-   Storing prediction history
-   Adding user authentication

------------------------------------------------------------------------

## 👨‍💻 Author

### Bansi Sojitra
------------------------------------------------------------------------

