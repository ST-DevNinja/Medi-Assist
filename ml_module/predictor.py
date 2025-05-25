import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Diabetes Model ----------
def load_diabetes_data():
    return pd.read_csv("data/diabetes.csv")

def train_diabetes_model():
    df = load_diabetes_data()
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Diabetes Model Accuracy: {accuracy:.2f}")
    
    joblib.dump(model, "ml_module/diabetes_model.pkl")
    return model

try:
    diabetes_model = joblib.load("ml_module/diabetes_model.pkl")
except FileNotFoundError:
    diabetes_model = train_diabetes_model()

def predict_diabetes(features: list) -> float:
    """
    Input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    Output: Probability of diabetes (0 to 1)
    """
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_df = pd.DataFrame([features], columns=column_names)
    prob = diabetes_model.predict_proba(input_df)[0][1]
    return round(prob, 3)


# ---------- General Model Configuration ----------
MODEL_CONFIG = {
    "lung": {
        "path": "data/survey lung cancer.csv",
        "target": "LUNG_CANCER",
        "binary_map": {"YES": 1, "NO": 0},
        "encoders": ["GENDER"],
        "features": None,  # auto-select
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
    }
}

# ---------- Model Training ----------
def train_model(kind: str, save_dir: str = "ml_module"):
    cfg = MODEL_CONFIG[kind]
    df = pd.read_csv(cfg["path"])
    
    if cfg.get("drop"):
        df = df.drop(columns=cfg["drop"])
    if cfg.get("dropna"):
        df = df.dropna()
    if cfg.get("derive"):
        df = cfg["derive"](df)
    if "binary_map" in cfg:
        df[cfg["target"]] = df[cfg["target"]].map(cfg["binary_map"])

    # Encode categorical features
    enc_cols = cfg.get("encoders") or df.select_dtypes(include=['object']).columns.tolist()
    le_dict = {}
    for col in enc_cols:
        if col in df.columns:
            le = LabelEncoder().fit(df[col])
            df[col] = le.transform(df[col])
            le_dict[col] = le

    # Features
    X = df[cfg["features"]] if cfg.get("features") else df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]

    # Scaling
    scaler = None
    if cfg.get("scale"):
        scaler = StandardScaler().fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if cfg.get("model_type") == "logistic":
        model = LogisticRegression(random_state=42, class_weight=cfg.get("class_weight"), max_iter=1000)
    else:
        model = RandomForestClassifier(random_state=42, class_weight=cfg.get("class_weight"))

    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, f"{save_dir}/{kind}_model.pkl")
    joblib.dump(le_dict, f"{save_dir}/{kind}_encoders.pkl")
    if scaler:
        joblib.dump(scaler, f"{save_dir}/{kind}_scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{save_dir}/{kind}_features.pkl")

    print(f"{kind.capitalize()} model trained and saved.")

# ---------- Prediction ----------
def predict(kind: str, inputs: list, save_dir: str = "ml_module") -> tuple:
    cfg = MODEL_CONFIG[kind]
    model = joblib.load(f"{save_dir}/{kind}_model.pkl")
    le_dict = joblib.load(f"{save_dir}/{kind}_encoders.pkl")
    scaler = joblib.load(f"{save_dir}/{kind}_scaler.pkl") if cfg.get("scale") else None
    feature_names = joblib.load(f"{save_dir}/{kind}_features.pkl")

    # Prepare input
    if kind == "skin":
        age, sex, loc, dx_type = inputs
        feat = [
            age,
            le_dict['sex'].transform([sex])[0],
            le_dict['localization'].transform([loc])[0],
            le_dict['dx_type'].transform([dx_type])[0]
        ]
    else:
        feat = inputs

    df_pred = pd.DataFrame([feat], columns=feature_names)
    
    if scaler:
        df_pred = pd.DataFrame(scaler.transform(df_pred), columns=feature_names)

    probs = model.predict_proba(df_pred)
    threshold = 0.53 if kind in ["prostate", "skin"] else 0.5
    pred = int(probs[0][1] >= threshold)
    
    return bool(pred), round(probs[0][1], 3)


