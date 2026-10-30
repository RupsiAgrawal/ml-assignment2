import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

st.set_page_config(page_title="Innings Tempo Classifier", page_icon="🏏", layout="wide")
st.title("🏏 T20 Innings Tempo Classifier")
st.caption("Each row is one innings. The saved models label it **high tempo (1)** or "
           "**low tempo (0)** from its composition: boundaries, dot balls, wickets, extras. "
           "Models are loaded from disk - nothing is retrained on the uploaded file.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
TARGET = open("model/target_name.txt").read().strip()
le = joblib.load("model/label_encoder.joblib")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (all features + target column)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"Target column expected: **{TARGET}**")
    try:
        with open("test_data.csv", "rb") as fh:
            st.download_button("Download sample test_data.csv", fh, file_name="test_data.csv",
                               mime="text/csv")
    except FileNotFoundError:
        pass

tab1, tab2 = st.tabs(["Selected model on your upload", "All five models (held-out matches)"])

with tab2:
    st.write("Metrics computed at training time on matches the models never saw.")
    table = pd.read_csv("model/metrics_table.csv")
    st.dataframe(table, use_container_width=True, hide_index=True)
    best = table.sort_values("MCC", ascending=False).iloc[0]
    st.success(f"Highest MCC: {best['ML Model Name']} ({best['MCC']:.4f})")

with tab1:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to begin (the sample file is downloadable there).")
    else:
        df = pd.read_csv(uploaded).dropna()
        pipe = joblib.load(MODEL_FILES[choice])
        needed = list(pipe.named_steps["pre"].feature_names_in_)
        missing = [c for c in needed + [TARGET] if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {', '.join(missing)}")
            st.stop()
        with st.expander(f"Preview of uploaded data ({len(df)} innings)"):
            st.dataframe(df.head(20), use_container_width=True)

        X = df[needed]
        y = le.transform(df[TARGET])
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        multi = len(le.classes_) > 2
        avg = "weighted" if multi else "binary"
        auc = (roc_auc_score(y, proba, multi_class="ovr", average="macro")
               if multi else roc_auc_score(y, proba[:, 1]))

        st.subheader(f"Evaluation metrics - {choice}")
        cols = st.columns(6)
        vals = [("Accuracy", accuracy_score(y, pred)), ("AUC", auc),
                ("Precision", precision_score(y, pred, average=avg, zero_division=0)),
                ("Recall", recall_score(y, pred, average=avg, zero_division=0)),
                ("F1", f1_score(y, pred, average=avg, zero_division=0)),
                ("MCC", matthews_corrcoef(y, pred))]
        for c, (n, v) in zip(cols, vals):
            c.metric(n, f"{v:.4f}")

        left, mid, right = st.columns(3)
        with left:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Greens",
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
        with mid:
            st.subheader("ROC curve")
            if multi:
                st.caption("ROC curve is shown for binary targets only.")
            else:
                fpr, tpr, _ = roc_curve(y, proba[:, 1])
                fig2, ax2 = plt.subplots(figsize=(4, 3.5))
                ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                ax2.plot([0, 1], [0, 1], "--", color="grey")
                ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
                ax2.legend()
                st.pyplot(fig2)
        with right:
            st.subheader("Classification report")
            rep = classification_report(y, pred, target_names=[str(c) for c in le.classes_],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
