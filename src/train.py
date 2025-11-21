import argparse, json, joblib, yaml, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- DAGSHUB + MLflow SETUP ---
USE_MLFLOW = False
try:
    import mlflow
    import mlflow.sklearn
    import dagshub

    # Inicializar conexión con DagsHub y MLflow
    dagshub.init(
        repo_owner='Lab-Min-Datos',
        repo_name='Proyecto_MLOps',
        mlflow=True
    )
    mlflow.autolog()  # autolog de sklearn (loggea métricas/modelo automáticamente)
    USE_MLFLOW = True
    print("MLflow activo (DagsHub).")
except Exception as e:
    print(f"No se pudo activar MLflow, entrenando sin tracking. Error: {e}")
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
        model = BernoulliNB(
            alpha=params["model"]["alpha"],
            fit_prior=params["model"]["fit_prior"]
        )
    else:
        raise ValueError("model.type no soportado")

    # -------- función auxiliar para no repetir código --------
    def _fit_and_eval(model):
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred)
        rec = recall_score(yte, pred)
        f1 = f1_score(yte, pred)

        # Guardar métricas localmente (para DVC)
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Guardar modelo localmente (DVC)
        joblib.dump(model, model_path)

        print(
            f"accuracy={acc:.3f}, precision={prec:.3f}, "
            f"recall={rec:.3f}, f1={f1:.3f}"
        )
        return model, metrics

    # --------- con MLflow: run explícito + log de hiperparámetros ---------
    if USE_MLFLOW:
        with mlflow.start_run():
            model, metrics = _fit_and_eval(model)

            # Hiperparámetros y config del split
            mlflow.log_params({
                "model_type": params["model"]["type"],
                "alpha": params["model"]["alpha"],
                "fit_prior": params["model"]["fit_prior"],
                "split_test_size": params["split"]["test_size"],
                "split_random_state": params["split"]["random_state"],
            })

            # Métricas (aunque autolog ya mete algunas, esto las garantiza como columnas claras)
            mlflow.log_metrics(metrics)

            # Modelo (autolog también lo suele registrar, pero esto es explícito)
            mlflow.sklearn.log_model(model, "model")

    # --------- sin MLflow: solo entrena y guarda para DVC ---------
    else:
        _fit_and_eval(model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    p = load_params(args.params)
    train(p, args.data, args.model, args.metrics)
