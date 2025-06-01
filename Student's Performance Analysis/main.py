#Load the dataset
import pandas as pd

df = pd.read_csv('student-mat.csv')

# Define the features and target variable
X = df.drop(['math score'], axis=1)
y = df['math score']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Now, let's do model training and evaluation
#1. linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)


#2. Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert grades to binary: pass (1) if G3 >= 10 else fail (0)
y_binary = (y >= 10).astype(int)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_bin, y_train_bin)
log_reg_pred = log_reg.predict(X_test_bin)
log_reg_acc = accuracy_score(y_test_bin, log_reg_pred)


#3. Decision Tree
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)


#   4. Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)


#5. Support Vector Machine
from sklearn.svm import SVR

svm = SVR()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_pred)

#Now. let's comapre the models based on Mean Squared Error (MSE) for regression models and accuracy for classification model
print(f"Linear Regression MSE: {lr_mse}")
print(f"Decision Tree MSE: {dt_mse}")
print(f"Random Forest MSE: {rf_mse}")
print(f"SVM MSE: {svm_mse}")
print(f"Logistic Regression Accuracy: {log_reg_acc}")

#Now let's create new student's data to predict their final grade
# Example new student data (ensure it matches the feature columns)
new_student = pd.DataFrame([X.iloc[0]])  # Using the first student's data as an example

# Predict using trained models
lr_new_pred = lr.predict(new_student)
dt_new_pred = dt.predict(new_student)
rf_new_pred = rf.predict(new_student)
svm_new_pred = svm.predict(new_student)

# For logistic regression, ensure the columns match those used in training
new_student_bin = new_student[X_train_bin.columns]  # Ensure same columns/order
log_reg_new_pred = log_reg.predict(new_student_bin)

print(f"Linear Regression Prediction: {lr_new_pred[0]}")
print(f"Decision Tree Prediction: {dt_new_pred[0]}")
print(f"Random Forest Prediction: {rf_new_pred[0]}")
print(f"SVM Prediction: {svm_new_pred[0]}")
print(f"Logistic Regression Prediction (Pass=1/Fail=0): {log_reg_new_pred[0]}")


# Save the trained models for future use 
import joblib
joblib.dump(lr, 'linear_regression_model.pkl')
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(dt, 'decision_tree_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
# Load the models back
lr_loaded = joblib.load('linear_regression_model.pkl')
log_reg_loaded = joblib.load('logistic_regression_model.pkl')
dt_loaded = joblib.load('decision_tree_model.pkl')
rf_loaded = joblib.load('random_forest_model.pkl')
svm_loaded = joblib.load('svm_model.pkl')
# Predict using loaded models
lr_loaded_pred = lr_loaded.predict(new_student)
dt_loaded_pred = dt_loaded.predict(new_student)
rf_loaded_pred = rf_loaded.predict(new_student)
svm_loaded_pred = svm_loaded.predict(new_student)
log_reg_loaded_pred = log_reg_loaded.predict(new_student_bin)  # Use new_student_bin here

print(f"Loaded Linear Regression Prediction: {lr_loaded_pred[0]}")
print(f"Loaded Decision Tree Prediction: {dt_loaded_pred[0]}")
print(f"Loaded Random Forest Prediction: {rf_loaded_pred[0]}")
print(f"Loaded SVM Prediction: {svm_loaded_pred[0]}")
print(f"Loaded Logistic Regression Prediction (Pass=1/Fail=0): {log_reg_loaded_pred[0]}")
# Save the predictions to a CSV file
predictions = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression'],
    'Prediction': [lr_new_pred[0], dt_new_pred[0], rf_new_pred[0], svm_new_pred[0], log_reg_new_pred[0]]
})
predictions.to_csv('student_predictions.csv', index=False)

