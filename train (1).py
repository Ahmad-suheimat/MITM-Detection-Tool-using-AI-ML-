import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------------------------------------------
# 1) Load Dataset
# ---------------------------------------------------
df = pd.read_excel("All_Labelled.xlsx")

DROP_COLS = [
    "application_name",
    "application_category_name",
    "application_is_guessed",
    "application_confidence",
    "requested_server_name",
    "client_fingerprint",
    "server_fingerprint",
    "user_agent",
    "content_type"
]

df.drop(columns=DROP_COLS, inplace=True)

# ---------------------------------------------------
# 2) Encode Label
# ---------------------------------------------------
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

# ---------------------------------------------------
# 3) Split features/labels
# ---------------------------------------------------
X = df.drop(columns=["Label"])
y = df["Label"]

# ---------------------------------------------------
# 4) Convert ALL columns to numeric
# ---------------------------------------------------

# Step A: Convert datetime to float ms
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.datetime64):
        X[col] = X[col].astype("int64") / 1e6

# Step B: Convert all string/object columns to numeric
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Step C: Replace NaN with zero
X = X.fillna(0)

# Step D: Ensure all columns are numeric
X = X.astype(float)

# ---------------------------------------------------
# 5) Scale Features
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# 6) Train/Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 7) Train RandomForest
# ---------------------------------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ---------------------------------------------------
# 8) Evaluate
# ---------------------------------------------------
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------
# 9) Save Model + Scaler + Label Encoder
# ---------------------------------------------------
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n[+] model.pkl created!")
print("[+] scaler.pkl created!")
print("[+] label_encoder.pkl created!")
