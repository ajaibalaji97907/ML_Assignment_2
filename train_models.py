import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



if not os.path.exists("model"):
    os.makedirs("model")



print("Loading dataset...")

df = pd.read_csv("heart_disease_uci.csv")

print("Original shape:", df.shape)




df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)



df = df.drop(["id"], axis=1)



df = df.dropna()

print("After cleaning shape:", df.shape)


categorical_cols = [
    "sex", "dataset", "cp",
    "fbs", "restecg",
    "exang", "slope", "thal"
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)



X = df.drop("num", axis=1)


joblib.dump(X.columns.tolist(), "model/training_columns.pkl")

y = df["num"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")



lr = LogisticRegression(max_iter=2000)
dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42
)



print("Training models...")

lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
knn.fit(X_train_scaled, y_train)
nb.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)



def evaluate_model(model, X_test_data, y_test):
    y_pred = model.predict(X_test_data)
    y_prob = model.predict_proba(X_test_data)[:, 1]

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }



print("Evaluating models...")

results = {}

results["Logistic Regression"] = evaluate_model(
    lr, X_test_scaled, y_test
)

results["Decision Tree"] = evaluate_model(
    dt, X_test, y_test
)

results["KNN"] = evaluate_model(
    knn, X_test_scaled, y_test
)

results["Naive Bayes"] = evaluate_model(
    nb, X_test_scaled, y_test
)

results["Random Forest"] = evaluate_model(
    rf, X_test, y_test
)

results["XGBoost"] = evaluate_model(
    xgb_model, X_test, y_test
)


results_df = pd.DataFrame(results).T

print("\nModel Comparison:")
print(results_df)

results_df.to_csv("model/model_comparison_results.csv")



print("Saving models...")

joblib.dump(lr, "model/logistic_model.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(nb, "model/naive_bayes_model.pkl")
joblib.dump(rf, "model/random_forest_model.pkl")
joblib.dump(xgb_model, "model/xgboost_model.pkl")

print("\n Training Completed Successfully!")



# Combine X_test and y_test
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df["num"] = y_test.values

# Save test data
X_test_df.to_csv("model/test_data.csv", index=False)

print("Test dataset saved as model/test_data.csv")

