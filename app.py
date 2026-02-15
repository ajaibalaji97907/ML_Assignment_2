import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


st.set_page_config(
    page_title="Heart Disease ML App",
    layout="wide"
)


st.title("Heart Disease Classification Dashboard")
st.write("Compare performance of 6 Machine Learning Classification Models")

st.markdown("---")


st.sidebar.title("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)



scaler = joblib.load("model/scaler.pkl")
training_columns = joblib.load("model/training_columns.pkl")
test_data = pd.read_csv("model/test_data.csv")



st.subheader("Download Test Dataset")

st.download_button(
    label="Download Test Data",
    data=test_data.to_csv(index=False),
    file_name="test_data.csv",
    mime="text/csv"
)

st.markdown("---")



st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    X = df.drop("num", axis=1)
    y = df["num"]

    
    for col in training_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[training_columns]



    if model_choice == "Logistic Regression":
        model = joblib.load("model/logistic_model.pkl")
        X_input = scaler.transform(X)

    elif model_choice == "Decision Tree":
        model = joblib.load("model/decision_tree.pkl")
        X_input = X

    elif model_choice == "KNN":
        model = joblib.load("model/knn_model.pkl")
        X_input = scaler.transform(X)

    elif model_choice == "Naive Bayes":
        model = joblib.load("model/naive_bayes_model.pkl")
        X_input = scaler.transform(X)

    elif model_choice == "Random Forest":
        model = joblib.load("model/random_forest_model.pkl")
        X_input = X

    elif model_choice == "XGBoost":
        model = joblib.load("model/xgboost_model.pkl")
        X_input = X

    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]



    st.markdown("---")
    st.header(f"{model_choice}")
    st.subheader("Model Evaluation Metrics")

    st.markdown("<br>", unsafe_allow_html=True)



    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    mcc = matthews_corrcoef(y, predictions)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("AUC Score", round(auc, 4))
    col3.metric("Precision", round(precision, 4))

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Recall", round(recall, 4))
    col2.metric("F1 Score", round(f1, 4))
    col3.metric("MCC", round(mcc, 4))

    st.markdown("---")


    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("---")


    st.subheader("Classification Report (Table View)")

    report_dict = classification_report(y, predictions, output_dict=True)
    
    report_df = pd.DataFrame(report_dict).transpose()

    report_df.rename(index={
        "0": "No Disease",
        "1": "Heart Disease"
    }, inplace=True)

    st.dataframe(report_df.style.format("{:.4f}"))

