import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


DATA_PATH = "../data/processed_pavement_data.csv"
MODEL_PATH = "../pavement_model.pkl"


# Load processed dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# Check required columns
required_columns = ["pci", "maintenance_priority"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# Input feature
X = df[["pci"]]

# Target variable
y = df["maintenance_priority"]


# Split data using the same settings as training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Load trained model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# Make predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Show some actual vs predicted values
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(results.head(20))
