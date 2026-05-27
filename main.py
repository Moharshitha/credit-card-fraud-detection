import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading data...")

# Load dataset
data = pd.read_csv("data.csv")

print("Data loaded successfully")

# Features and Labels
X = data.drop("Class", axis=1)
y = data["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")

# Model
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(X_train, y_train)

print("Model trained")

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# Test with a sample transaction
sample = X_test.iloc[[0]]

result = model.predict(sample)

if result[0] == 0:
    print("Not Fraud ✅")
else:
    print("Fraud Transaction 🚨")