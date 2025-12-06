# src/evaluate.py

import json
import joblib
import yaml
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def split_data(df, params):
    """Reproduce el mismo split que train.py para obtener el mismo test set."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["churn"])
    y = df["churn"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y
    )
    return Xte, yte


def evaluate(model_path, data_path, metrics_out, roc_out):
    params = load_params()

    df = pd.read_csv(data_path)
    Xte, yte = split_data(df, params)

    model = joblib.load(model_path)

    # Predicción
    pred = model.predict(Xte)

    # Probabilidades
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)[:, 1]
    else:
        proba = None

    # Métricas
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
    }

    # ROC AUC
    if proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(yte, proba))
        except Exception:
            metrics["roc_auc"] = None

    # Guardar métricas
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    # Curva ROC
    if proba is not None:
        fpr, tpr, _ = roc_curve(yte, proba)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC")
        plt.legend()
        plt.savefig(roc_out, dpi=120, bbox_inches="tight")
        plt.close()

    print("Evaluación completada.")
    print("Métricas guardadas en:", metrics_out)
    print("Curva ROC guardada en:", roc_out)


if __name__ == "__main__":
    evaluate(
        model_path="models/churn_model.pkl",
        data_path="data/processed/telco_clean.csv",
        metrics_out="reports/metrics_final.json",
        roc_out="reports/roc_curve.png"
    )
