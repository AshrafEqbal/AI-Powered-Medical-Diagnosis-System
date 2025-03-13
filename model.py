import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_DIR = "Datasets"
MODEL_DIR = "Models"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model(X, y, model_name, use_svm=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='linear') if use_svm else LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name.capitalize()} Model Accuracy: {accuracy:.2f}")

    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.sav")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"{model_name.capitalize()} model saved successfully at {model_path}!\n")

try:
    diabetes_data = pd.read_csv(os.path.join(DATASET_DIR, "diabetes.csv"))
    X_diabetes = diabetes_data.iloc[:, :-1]
    y_diabetes = diabetes_data.iloc[:, -1]
    train_and_save_model(X_diabetes, y_diabetes, "diabetes")

    heart_data = pd.read_csv(os.path.join(DATASET_DIR, "heart_disease_data.csv"))
    X_heart = heart_data.iloc[:, :-1]
    y_heart = heart_data.iloc[:, -1]
    train_and_save_model(X_heart, y_heart, "heart_disease")

    parkinsons_data = pd.read_csv(os.path.join(DATASET_DIR, "parkinson_data.csv"))
    X_parkinsons = parkinsons_data.drop(columns=["name", "status"], errors="ignore")
    y_parkinsons = parkinsons_data["status"]
    train_and_save_model(X_parkinsons, y_parkinsons, "parkinsons", use_svm=True)

    lung_cancer_data = pd.read_csv(os.path.join(DATASET_DIR, "survey lung cancer.csv"))
    lung_cancer_data["GENDER"] = lung_cancer_data["GENDER"].map({"M": 1, "F": 0})
    X_lung_cancer = lung_cancer_data.drop(columns=["LUNG_CANCER"], errors="ignore")
    y_lung_cancer = lung_cancer_data["LUNG_CANCER"]
    train_and_save_model(X_lung_cancer, y_lung_cancer, "lungs_disease")

    print("All models trained and saved successfully!")

except FileNotFoundError as e:
    print(f"Error: {e}. Check if the dataset files exist in the 'Datasets' folder.")
