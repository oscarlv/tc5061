"""Entrenar un modelo Extra Trees y guardar la corrida en MLflow."""
import argparse
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "TC5061_Semana4_ExtraTrees"


def max_features_value(value):
    if value in ("sqrt", "log2"):
        return value
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Usa sqrt, log2 o una fracción entre 0 y 1.") from error
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("La fracción debe ser mayor que 0 y menor o igual que 1.")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-estimators", type=int, default=300, help="Número de árboles.")
    parser.add_argument("--max-features", type=max_features_value, default="sqrt",
                        help="Variables consideradas en cada división: sqrt, log2 o fracción.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Muestras mínimas en una hoja.")
    parser.add_argument("--max-depth", type=int, default=None, help="Profundidad máxima; sin argumento no se limita.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para repetir la partición y el entrenamiento.")
    parser.add_argument("--run-name", default="01_referencia", help="Nombre de la corrida en MLflow.")
    args = parser.parse_args()
    if args.n_estimators < 1 or args.min_samples_leaf < 1 or (args.max_depth is not None and args.max_depth < 1):
        parser.error("Los tamaños del modelo deben ser positivos.")
    if not 0 <= args.seed < 2**32:
        parser.error("La semilla debe estar entre 0 y 2**32 - 1.")
    return args


def load_data():
    red = pd.read_csv(ROOT / "data/winequality-red.csv", sep=";")
    white = pd.read_csv(ROOT / "data/winequality-white.csv", sep=";")
    red["tipo"] = "tinto"
    white["tipo"] = "blanco"
    return pd.concat([red, white], ignore_index=True)


def prepare_data(data, seed):
    # Conservamos la limpieza y la definición del objetivo de la semana 3.
    measurements = data.drop(columns=["quality", "tipo"])
    if data.isna().any().any() or not np.isfinite(measurements.to_numpy()).all():
        raise ValueError("Los datos tienen valores faltantes o no finitos.")
    if (measurements < 0).any().any() or not data["pH"].between(0, 14).all():
        raise ValueError("Hay mediciones fuera de los rangos esperados.")
    if not data["quality"].between(0, 10).all() or (data["quality"] % 1 != 0).any():
        raise ValueError("La calidad debe ser un entero entre 0 y 10.")
    if (data["free sulfur dioxide"] > data["total sulfur dioxide"]).any():
        raise ValueError("El dióxido de azufre libre supera al total.")
    clean = data.drop_duplicates().reset_index(drop=True).copy()
    if clean.drop(columns="quality").duplicated().any():
        raise ValueError("Hay filas con las mismas variables y distinta calidad.")
    clean["tipo"] = clean["tipo"].map({"tinto": 0, "blanco": 1})
    X = clean.drop(columns="quality")
    y = (clean["quality"] >= 7).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def train_model(args, X_train, y_train):
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, max_depth=args.max_depth,
        random_state=args.seed, n_jobs=1)
    # Comparamos configuraciones con datos de entrenamiento, sin elegir por la prueba.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc", error_score="raise")
    model.fit(X_train, y_train)
    return model, scores


def evaluate_model(model, X_test, y_test, scores):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "cv_roc_auc": float(scores.mean()), "cv_roc_auc_std": float(scores.std()),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
    }
    return metrics, predictions, probabilities


def save_artifacts(folder, X_test, y_test, predictions, probabilities):
    pd.DataFrame({"indice": X_test.index, "real": y_test.to_numpy(),
                  "prediccion": predictions, "probabilidad_alta": probabilities}).to_csv(
                      folder / "predicciones.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Resto", "Calidad alta"], ax=ax, colorbar=False)
    ax.set_title("Resultados en el conjunto de prueba")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(folder / "matriz_confusion.png")
    plt.close(fig)
    for path in folder.iterdir():
        mlflow.log_artifact(str(path))


def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = prepare_data(load_data(), args.seed)
    (ROOT / ".mlflow").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / '.mlflow/week4.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name) as run:
        parameters = {"n_estimators": args.n_estimators, "max_features": args.max_features,
                      "min_samples_leaf": args.min_samples_leaf, "max_depth": args.max_depth,
                      "seed": args.seed, "test_size": 0.2, "cv_folds": 5,
                      "n_train": len(X_train), "n_test": len(X_test), "threshold": 0.5}
        for name, value in parameters.items():
            mlflow.log_param(name, value)
        model, scores = train_model(args, X_train, y_train)
        metrics, predictions, probabilities = evaluate_model(model, X_test, y_test, scores)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        with tempfile.TemporaryDirectory() as temporary:
            save_artifacts(Path(temporary), X_test, y_test, predictions, probabilities)
        example = X_train.head(5)
        mlflow.sklearn.log_model(model, name="modelo", input_example=example,
                                signature=infer_signature(example, model.predict(example)),
                                skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f"Corrida guardada: {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
