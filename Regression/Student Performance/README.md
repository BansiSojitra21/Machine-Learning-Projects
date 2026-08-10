# 📊 Student Performance Prediction

A Machine Learning regression project that predicts a student's **Performance Index** based on academic and personal factors such as study hours, previous scores, sleep duration, practice papers, and extracurricular activities.

The project compares multiple regression algorithms and evaluates them not only on test data but also on a separate **unseen dataset**.

---

## 🎯 Project Objective

The objective of this project is to build a machine learning model that can predict a student's **Performance Index** and determine which regression algorithm performs best on both test and unseen data.

The project follows a complete machine learning workflow:

**Data Loading → Data Understanding → Data Cleaning → EDA → Feature Engineering → Preprocessing → Train/Test/Unseen Split → Model Training → Evaluation → Model Saving**

---

## 📁 Dataset

The project uses `Student_Performance.csv`.

### Features

| Feature                            | Description                                                    |
| ---------------------------------- | -------------------------------------------------------------- |
| `Hours Studied`                    | Number of hours studied                                        |
| `Previous Scores`                  | Student's previous academic score                              |
| `Extracurricular Activities`       | Whether the student participates in extracurricular activities |
| `Sleep Hours`                      | Average hours of sleep                                         |
| `Sample Question Papers Practiced` | Number of sample papers practiced                              |
| `Performance Index`                | Target variable representing student performance               |

### Target

**`Performance Index`**

This is a **regression problem** because the target variable is continuous.

---

## 🔍 Exploratory Data Analysis

The notebook performs several exploratory data analysis steps:

* Dataset shape and information
* Descriptive statistics
* Missing-value detection
* Duplicate detection and removal
* Distribution of extracurricular activities
* Relationship between hours studied and performance
* Correlation analysis
* Correlation heatmap
* Box plots for numerical features to inspect potential outliers

### Example Insight

The project visualizes the relationship between:

**Hours Studied → Performance Index**

to understand how study time relates to student performance.

---

## 🧹 Data Preprocessing

### 1. Missing Values

### 2. Duplicate Removal

### 3. Categorical Encoding

### 4. Removing Unused Feature

### 5. Feature Scaling and Encoding

---

## ✂️ Data Splitting Strategy

A separate portion of the original dataset is reserved as **unseen data**.

### Step 1 — Unseen Data

```text
90% → Remaining Dataset
10% → Unseen Dataset
```

### Step 2 — Training and Testing

The remaining 90% is further divided:

```text
80% → Training Data
20% → Testing Data
```

This gives the project three separate datasets:

```text
Original Dataset
       │
       ├── 10% → Unseen Data
       │
       └── 90% → Remaining Data
                    │
                    ├── 80% → Training
                    │
                    └── 20% → Testing
```

The unseen dataset is kept separate until the final evaluation.

---

## 🤖 Machine Learning Models

Five regression algorithms are trained and compared:

### 1. Linear Regression

### 2. Decision Tree Regressor

### 3. Random Forest Regressor

### 4. XGBoost Regressor

### 5. Support Vector Regression
---

## 📏 Evaluation Metrics

The models are evaluated using:

### R² Score

Measures how well the model explains the variation in the target variable.

Higher is better.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

Lower is better.

### RMSE — Root Mean Squared Error

Measures prediction error while giving more weight to larger errors.

Lower is better.

> **Note:** Since this is a regression problem, the reported "Accuracy" in the notebook is actually the **R² Score** returned through `r2_score()`.

---

## 🏆 Model Selection

The models are compared using:

* Test R² Score
* Unseen R² Score
* Unseen MAE
* Unseen RMSE

---

## 💾 Model Saving

The trained SVR model is saved using Pickle:

```python
with open("student_performance.pkl", "wb") as file:
    pickle.dump(svr, file)
```

The preprocessing object is also saved:

```python
with open("preprocessor.pkl", "wb") as file:
    pickle.dump(preprocessor, file)
```

This allows the trained model and preprocessing steps to be reused later without retraining from scratch.

---

## 📂 Project Structure

```text
Student-Performance-Prediction/
│
├── Student Performance Regression.ipynb
├── Student_Performance.csv
├── student_performance.pkl
├── preprocessor.pkl
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Pickle

### Machine Learning

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* XGBoost Regression
* Support Vector Regression (SVR)

### Preprocessing

* StandardScaler
* OneHotEncoder
* ColumnTransformer

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/student-performance-prediction.git
```

### 2. Navigate to the Project

```bash
cd student-performance-prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Student Performance Regression.ipynb
```

### 5. Run the Notebook

Run the cells sequentially to:

* Load the dataset
* Perform EDA
* Clean the data
* Preprocess features
* Train models
* Compare models
* Evaluate unseen data
* Save the final model

---

## 📌 Key Learning Outcomes

Through this project, I practiced:

* Data exploration and understanding
* Data cleaning
* Duplicate handling
* Categorical feature encoding
* Feature scaling
* `ColumnTransformer`
* Train/Test splitting
* Creating a separate unseen dataset
* Regression model comparison
* Model evaluation using R², MAE and RMSE
* Model persistence using Pickle
* Selecting a final model based on unseen-data performance

---

## 🔮 Future Improvements

Possible improvements for this project include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Cross-validation
* Feature engineering
* Comparing additional regression algorithms
* Building a prediction interface using Streamlit
* Creating a complete preprocessing + model `Pipeline`
* Deploying the trained model as a web application

---

## 👨‍💻 Author

**Bansi Sojitra**

Data Science / Machine Learning Enthusiast

---

If you find this project helpful for learning Machine Learning, consider giving the repository a ⭐ on GitHub.
