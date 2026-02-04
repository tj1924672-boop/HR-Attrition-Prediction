import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv(r"C:\Users\USER\AppData\Local\Temp\4769bb7c-984a-4373-a1ed-70bdda1af55f_archive (5).zip.55f\WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(data)
# Show first 5 rows
print(data.head())

# Dataset info
print(data.info())

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include='object'):
    data[col] = le.fit_transform(data[col])

# Split data
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Attrition count
sns.countplot(x="Attrition", data=data)
plt.title("Employee Attrition Count")
plt.show()
