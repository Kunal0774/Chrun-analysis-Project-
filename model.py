# =========================
# CUSTOMER CHURN MODEL
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
df = pd.read_csv("churn_data.csv")

print("First 5 rows:")
print(df.head())

# Step 2: Data Info
print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Data Cleaning
df.fillna(method='ffill', inplace=True)

# Step 4: Convert Categorical to Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# Step 5: EDA (Simple Graph)
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Step 6: Split Data
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 7: Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Save Model
import pickle

pickle.dump(model, open("churn_model.pkl", "wb"))

print("\nModel saved as churn_model.pkl")