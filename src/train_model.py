import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/cleaned_pavement_data.csv"
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


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)
