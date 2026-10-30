"""Train 5 classifiers on one dataset and save models, test data and metrics.

Usage:  python train.py --data innings_dataset.csv --target high_tempo --group match_id
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def compute_metrics(y_true, y_pred, y_proba):
    n_classes = len(set(y_true))
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    avg = "binary" if n_classes == 2 else "weighted"
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--group", default=None, help="column to keep together in the split (dropped afterwards)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna().reset_index(drop=True)
    n_feat = df.shape[1] - 1 - (1 if args.group else 0)
    print(f"Rows: {len(df)}, features: {n_feat}")

    le = LabelEncoder()
    y = le.fit_transform(df[args.target])
    X = df.drop(columns=[args.target])

    if args.group:
        # keep every row of one group (e.g. a match) on the same side of the split
        groups = X[args.group]
        X = X.drop(columns=[args.group])
        tr_idx, te_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed).split(X, y, groups))
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    def make_pre():
        return ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=args.seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=args.seed),
    }

    os.makedirs("model", exist_ok=True)
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        m = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te))
        rows.append({"ML Model Name": name, **m})
        joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")
        print(name, {k: round(v, 4) for k, v in m.items()})

    joblib.dump(le, "model/label_encoder.joblib")
    pd.DataFrame(rows).round(4).to_csv("model/metrics_table.csv", index=False)

    test_df = X_te.copy()
    test_df[args.target] = le.inverse_transform(y_te)
    test_df.to_csv("test_data.csv", index=False)
    with open("model/target_name.txt", "w") as f:
        f.write(args.target)
    print("\nSaved models, model/metrics_table.csv and test_data.csv")


if __name__ == "__main__":
    main()
