# train.py - Drug Classification Model Training Pipeline
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Create necessary directories
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

print("Loading and preprocessing data...")

# Load the dataset
drug_df = pd.read_csv("data/drug200.csv")
drug_df = drug_df.sample(frac=1, random_state=125)
print(f"Dataset shape: {drug_df.shape}")
print("First 3 rows:")
print(drug_df.head(3))

# Prepare features and target
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Define preprocessing pipeline
cat_col = [1, 2, 3]  # Sex, BP, Cholesterol
num_col = [0, 4]     # Age, Na_to_K

transform = ColumnTransformer([
    ("encoder", OrdinalEncoder(), cat_col),
    ("num_imputer", SimpleImputer(strategy="median"), num_col),
    ("num_scaler", StandardScaler(), num_col),
])

# Create complete pipeline
pipe = Pipeline(steps=[
    ("preprocessing", transform),
    ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
])

print("Training model...")
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy, 2) * 100}%")
print(f"F1 Score: {round(f1, 2)}")

# Save metrics to file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}")

print("Generating confusion matrix...")

# Create and save confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.title("Drug Classification Confusion Matrix")
plt.savefig("Results/model_results.png", dpi=120, bbox_inches='tight')
plt.close()

# Save the model
print("Saving model...")
joblib.dump(pipe, "Model/drug_pipeline.joblib")
print("Training complete - model saved to Model/drug_pipeline.joblib")

# Verify model can be loaded
pipe_loaded = joblib.load("Model/drug_pipeline.joblib")
print("Model verification: Successfully loaded saved model")