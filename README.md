# ML-Projects
# Student's Performance Analysis

This project analyzes students' academic performance using machine learning algorithms. The dataset used contains various features such as demographics, study time, family support, etc., and aims to predict math scores and pass/fail classification.

## 📊 Dataset
The dataset used is `student-mat.csv`, originally sourced from Kaggle. It includes information about students’ backgrounds and their academic scores.

## 🔍 Objectives

- Predict final math grades using regression models.
- Predict pass/fail status using classification.
- Compare different models based on performance.
- Save and load trained models using `joblib`.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib

## 📈 Models Trained

### Regression Models (for predicting Math Score):
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor

### Classification Model (for predicting Pass/Fail):
- Logistic Regression

## 📌 Model Evaluation

- Regression models were evaluated using **Mean Squared Error (MSE)**.
- Logistic Regression was evaluated using **Accuracy**.

## 📦 Outputs

- Trained models are saved as `.pkl` files using `joblib`.
- Predictions for a new student are saved in `student_predictions.csv`.

## 💾 Files in this Project

- `Student's Performance Analysis.ipynb` – The main code notebook
- `student-mat.csv` – The dataset
- `student_predictions.csv` – Output predictions
- `.pkl` model files – Saved ML models
- `README.md` – Project documentation

## 🔄 Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Visual analysis of feature importance
- Use cross-validation for model reliability

---

> Created with ❤️ as part of the **ML Projects** repository.
