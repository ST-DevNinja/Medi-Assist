import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ---------- Diabetes Model ----------
def load_diabetes_data():
    return pd.read_csv("data/diabetes.csv")

def train_diabetes_model(save_dir="ml_module"):
    df = load_diabetes_data()
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Diabetes Model Accuracy: {accuracy:.2f}")

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, f"{save_dir}/diabetes_model.pkl")
    joblib.dump(accuracy, f"{save_dir}/diabetes_accuracy.pkl")
    return model, accuracy

try:
    diabetes_model = joblib.load("ml_module/diabetes_model.pkl")
    diabetes_accuracy = joblib.load("ml_module/diabetes_accuracy.pkl")
except FileNotFoundError:
    diabetes_model, diabetes_accuracy = train_diabetes_model()

def predict_diabetes(features: list) -> tuple:
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_df = pd.DataFrame([features], columns=column_names)
    prob = diabetes_model.predict_proba(input_df)[0][1]
    return round(prob, 3), round(diabetes_accuracy, 3)

# ---------- Model Configurations ----------
MODEL_CONFIG = {
    "lung": {
        "path": "data/survey lung cancer.csv",
        "target": "LUNG_CANCER",
        "binary_map": {"YES": 1, "NO": 0},
        "encoders": ["GENDER"],
        "features": None,
        "scale": False,
        "class_weight": "balanced",
        "model_type": "rf"
    },
    "prostate": {
        "path": "data/prostate_cancer_prediction.csv",
        "drop": ["Patient_ID"],
        "target": "Early_Detection",
        "features": ["Age", "PSA_Level", "BMI", "Prostate_Volume"],
        "scale": True,
        "class_weight": "balanced",
        "model_type": "logistic"
    },
    "skin": {
        "path": "data/HAM10000_metadata.csv",
        "dropna": True,
        "derive": lambda df: df.assign(
            cancer=df['dx'].apply(lambda x: 0 if x == 'nv' else 1)
        ),
        "target": "cancer",
        "encoders": ['sex', 'localization', 'dx_type'],
        "features": ['age', 'sex', 'localization', 'dx_type'],
        "scale": False,
        "class_weight": "balanced",
        "model_type": "rf"
    },
    "heart": {
        "path": "data/heart.csv",
        "target": "target",
        "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        "scale": True,
        "class_weight": "balanced",
        "model_type": "logistic"
    },
    "kidney": {
        "path": "data/kidney.csv",
        "target": "classification",
        "dropna": True,
        "binary_map": {"ckd": 1, "notckd": 0},
        "encoders": ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
        "features": ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot',
                     'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'],
        "scale": True,
        "class_weight": "balanced",
        "model_type": "rf"
    },
    "liver": {
        "path": "data/indian_liver_patient.csv",
        "target": "Dataset",
        "features": ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                     'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                     'Albumin', 'Albumin_and_Globulin_Ratio'],
        "binary_map": {1: 1, 2: 0},
        "encoders": ['Gender'],
        "dropna": True,
        "scale": True,
        "class_weight": "balanced",
        "model_type": "rf"
    },
    "asthma": {
        "path": "data/asthma.csv",
        "target": "Severity_None",
        "features": [
            "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton",
            "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing",
            "Age_0-9", "Age_10-19", "Age_20-24", "Age_25-59", "Age_60+",
            "Gender_Female", "Gender_Male",
            "Severity_Mild", "Severity_Moderate"
        ],
        "scale": False,
        "class_weight": "balanced",
        "model_type": "rf"
    }
}

# ---------- Model Trainer ----------
def train_model(kind: str, save_dir: str = "ml_module"):
    cfg = MODEL_CONFIG[kind]
    df = pd.read_csv(cfg["path"])

    if cfg.get("drop"):
        df = df.drop(columns=cfg["drop"])
    if cfg.get("dropna"):
        df = df.dropna()
    if cfg.get("derive"):
        df = cfg["derive"](df)
    if cfg.get("binary_map"):
        df[cfg["target"]] = df[cfg["target"]].map(cfg["binary_map"])

    le_dict = {}
    for col in cfg.get("encoders", []):
        if col in df.columns:
            le = LabelEncoder().fit(df[col])
            df[col] = le.transform(df[col])
            le_dict[col] = le

    X = df[cfg["features"]] if cfg.get("features") else df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]

    scaler = None
    if cfg.get("scale"):
        scaler = StandardScaler().fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if cfg["model_type"] == "logistic":
        model = LogisticRegression(random_state=42, class_weight=cfg["class_weight"], max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42, class_weight=cfg["class_weight"])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{kind.capitalize()} Model Accuracy: {accuracy:.2f}")

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, f"{save_dir}/{kind}_model.pkl")
    joblib.dump(le_dict, f"{save_dir}/{kind}_encoders.pkl")
    if scaler:
        joblib.dump(scaler, f"{save_dir}/{kind}_scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{save_dir}/{kind}_features.pkl")
    joblib.dump(accuracy, f"{save_dir}/{kind}_accuracy.pkl")

    print(f"{kind.capitalize()} model trained and saved.")

# ---------- Generic Prediction ----------
def predict(kind: str, inputs: list, save_dir: str = "ml_module") -> tuple:
    cfg = MODEL_CONFIG[kind]
    model = joblib.load(f"{save_dir}/{kind}_model.pkl")
    le_dict = joblib.load(f"{save_dir}/{kind}_encoders.pkl")
    scaler = joblib.load(f"{save_dir}/{kind}_scaler.pkl") if cfg.get("scale") else None
    feature_names = joblib.load(f"{save_dir}/{kind}_features.pkl")
    accuracy = joblib.load(f"{save_dir}/{kind}_accuracy.pkl")

    feat = inputs.copy()

    for i, col in enumerate(feature_names):
        if col in le_dict:
            le = le_dict[col]
            try:
                feat[i] = le.transform([feat[i]])[0]
            except ValueError:
                print(f"Warning: unseen label '{feat[i]}' for column '{col}'. Using default encoding.")
                feat[i] = 0

    df_pred = pd.DataFrame([feat], columns=feature_names)

    if scaler:
        df_pred = pd.DataFrame(scaler.transform(df_pred), columns=feature_names)

    probs = model.predict_proba(df_pred)
    pred = int(probs[0][1] >= 0.5)

    return bool(pred), round(probs[0][1], 3), round(accuracy, 3)

# ---------- Specific Predictors ----------
def predict_heart(features: list) -> tuple:
    return predict("heart", features)

def predict_asthma(features: list) -> tuple:
    return predict("asthma", features)
