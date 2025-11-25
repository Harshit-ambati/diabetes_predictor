import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("diabetes_data.csv")

# 1. Drop rows with any missing values (NaN) to prevent training errors
print(f"Original dataset size: {len(data)} rows.")
data = data.dropna()
print(f"Cleaned dataset size: {len(data)} rows.")

# 2. Convert 'FamilyHistory' from 'Yes'/'No' to 1/0 
data['FamilyHistory'] = data['FamilyHistory'].apply(lambda x: 1 if x == 'Yes' else 0)

X = data[['Age', 'BMI', 'Glucose', 'BloodPressure', 'FamilyHistory']]
y = data['Diabetic'] # Target

y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

log_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))
tree_acc = accuracy_score(y_test, dtree.predict(X_test))

print(f"✅ Logistic Regression Accuracy: {log_acc:.2f}")
print(f"✅ Decision Tree Accuracy: {tree_acc:.2f}")


joblib.dump(log_reg, "logistic_model.pkl")
joblib.dump(dtree, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models and scaler saved successfully!")