import argparse, json, joblib, yaml, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

USE_MLFLOW = False
try:
    import mlflow
    if os.getenv("MLFLOW_TRACKING_URI"):
        USE_MLFLOW = True
except Exception:
    USE_MLFLOW = False

def load_params(pfile):
    with open(pfile) as f:
        return yaml.safe_load(f)

def train(params, data_path, model_path, metrics_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y
    )

    if params["model"]["type"] == "BernoulliNB":
        model = BernoulliNB(alpha=params["model"]["alpha"], fit_prior=params["model"]["fit_prior"])
    else:
        raise ValueError("model.type no soportado")

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred)
    rec = recall_score(yte, pred)
    f1 = f1_score(yte, pred)
    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump({"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}, f, indent=2)
    print(f"accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    
    p = load_params(args.params)
    train(p, args.data, args.model, args.metrics)