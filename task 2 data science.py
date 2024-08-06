import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Loading and exploring the data
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_data = pd.read_csv(url)
print(titanic_data.head())

# Step 2: Processing data
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
columns_to_drop = [col for col in columns_to_drop if col in titanic_data.columns]
titanic_data = titanic_data.drop(columns=columns_to_drop, axis=1)

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

# Step 3: Feature Scaling
scaler = StandardScaler()
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
titanic_data[numerical_cols] = scaler.fit_transform(titanic_data[numerical_cols])

# Step 4: Split Data into Training and Testing Sets
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Predictive Model 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
sample_passenger = {
    'Pclass': 3,
    'Age': 25,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.5,
    'Sex_male': 1,  
    'Embarked_Q': 0,
    'Embarked_S': 1
}
input_data = pd.DataFrame([sample_passenger])
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])  # Scale numerical features
prediction = model.predict(input_data)

print(f"Predicted Survival Outcome: {prediction[0]}")