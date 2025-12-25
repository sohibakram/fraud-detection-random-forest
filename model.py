import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Clean column names (safety)
df.columns = df.columns.str.strip()

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("Number of features:", X.shape[1])   # MUST be 30
print("Feature columns:\n", X.columns)

# 5. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 6. Train Random Forest (handle imbalance)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# 8. Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as fraud_model.pkl")

