import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report

DATA_PATH = Path("credit_card_fraud_balanced.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError("Place credit_card_fraud_balanced.csv beside this script.")

df = pd.read_csv(DATA_PATH)
required = {"TransactionID", "TransactionDate", "Amount", "MerchantID", "TransactionType", "Location", "IsFraud"}
if required - set(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

def make_features(frame):
    x = frame.copy()
    dt = pd.to_datetime(x["TransactionDate"], errors="coerce")
    x["TransactionHour"] = dt.dt.hour
    x["TransactionDayOfWeek"] = dt.dt.dayofweek
    x["TransactionMonth"] = dt.dt.month
    x["TransactionDay"] = dt.dt.day
    return x.drop(columns=["TransactionID", "TransactionDate"], errors="ignore")

X = make_features(df.drop(columns=["IsFraud"]))
y = df["IsFraud"].astype(int)
numeric = ["Amount", "MerchantID", "TransactionHour", "TransactionDayOfWeek", "TransactionMonth", "TransactionDay"]
categorical = ["TransactionType", "Location"]
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
model = Pipeline([
    ("preprocess", prep),
    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "average_precision": float(average_precision_score(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}
joblib.dump(model, "fraud_model.joblib")
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(classification_report(y_test, pred, zero_division=0))
print("Saved fraud_model.joblib and metrics.json")
