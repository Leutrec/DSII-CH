"""Clasificador supervisado de AG News con TF-IDF y regresión logística.

El vectorizador se ajusta solo con el train. El test se transforma al final,
una sola vez, para no filtrar su vocabulario al modelo.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from cargar_dataset import cargar  # noqa: E402
from src.eda_ag_news import CLASS_ORDER, content_stop_words  # noqa: E402
from src.preprocess import preprocess_corpus  # noqa: E402

# Carpeta versionada: outputs/ está en .gitignore y no llega a GitHub.
OUTPUT_DIR = ROOT / "resultados" / "tfidf"

# Comparación acotada al train. El test no participa de esta elección.
EXPERIMENTS = [
    {"max_features": 5_000, "ngram_range": (1, 1)},
    {"max_features": 10_000, "ngram_range": (1, 1)},
    {"max_features": 10_000, "ngram_range": (1, 2)},
    {"max_features": 20_000, "ngram_range": (1, 2)},
]


def build_pipeline(max_features: int, ngram_range: tuple[int, int], stop_words: list[str]) -> Pipeline:
    """TF-IDF y clasificador en un solo objeto, para que cada fold reajuste el vocabulario."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    token_pattern=r"(?u)\b[a-z]{2,}\b",
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    stop_words=stop_words,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def select_on_train(texts: list[str], labels: np.ndarray, stop_words: list[str]) -> tuple[dict, list[dict]]:
    """Elige max_features y ngram_range con F1-macro en validación cruzada del train."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for config in EXPERIMENTS:
        pipeline = build_pipeline(config["max_features"], config["ngram_range"], stop_words)
        scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")
        row = {
            "max_features": config["max_features"],
            "ngram_range": list(config["ngram_range"]),
            "f1_macro_mean": float(scores.mean()),
            "f1_macro_std": float(scores.std()),
            "folds": [float(score) for score in scores],
        }
        rows.append(row)
        ngram = f"{config['ngram_range'][0]}-{config['ngram_range'][1]}"
        print(
            f"CV  max_features={config['max_features']:<5}  "
            f"ngram={ngram}  F1-macro={scores.mean():.4f} ± {scores.std():.4f}"
        )
    best = max(rows, key=lambda row: row["f1_macro_mean"])
    return best, rows


def top_coefficients(pipeline: Pipeline, k: int = 8) -> dict[str, list[dict]]:
    """Los n-gramas con mayor peso positivo en cada clase."""
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    classifier = pipeline.named_steps["clf"]
    by_class = {}
    for index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[index]
        order = np.argsort(coefficients)[-k:][::-1]
        by_class[str(label)] = [
            {"term": str(names[term_index]), "coef": float(coefficients[term_index])}
            for term_index in order
        ]
    return by_class


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, path: Path) -> np.ndarray:
    matrix = confusion_matrix(labels, predictions, labels=CLASS_ORDER)
    figure, axis = plt.subplots(figsize=(7.2, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axis.set_xlabel("Predicción")
    axis.set_ylabel("Clase real")
    axis.set_title("Matriz de confusión — test de AG News")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "#1a1a1a"
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return matrix


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = cargar("ag_news")

    # La lematización es local a cada documento: no usa conteos del otro split.
    print("Preprocesando train...")
    train_text = preprocess_corpus(train_df["text"].fillna("").tolist())
    print("Preprocesando test...")
    test_text = preprocess_corpus(test_df["text"].fillna("").tolist())
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    stop_words = content_stop_words()

    best, experiments = select_on_train(train_text, y_train, stop_words)
    ngram_range = tuple(best["ngram_range"])
    pipeline = build_pipeline(best["max_features"], ngram_range, stop_words)

    # fit_transform queda adentro de Pipeline.fit y solo ve el train.
    pipeline.fit(train_text, y_train)
    x_train = pipeline.named_steps["tfidf"].transform(train_text)
    x_test = pipeline.named_steps["tfidf"].transform(test_text)
    y_pred = pipeline.predict(test_text)

    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, digits=4, zero_division=0)
    matrix = save_confusion_matrix(y_test, y_pred, OUTPUT_DIR / "matriz_confusion.png")
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    per_class = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "dataset": "ag_news",
        "model": "LogisticRegression",
        "solver": "lbfgs",
        "C": 1.0,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "stop_words": len(stop_words),
        "vectorizer": {
            "max_features": best["max_features"],
            "ngram_range": best["ngram_range"],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "vocabulary_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        },
        "matrix_shape": {"train": list(x_train.shape), "test": list(x_test.shape)},
        "cv_experiments": experiments,
        "selected_by": "f1_macro medio en StratifiedKFold del train",
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", labels=CLASS_ORDER)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", labels=CLASS_ORDER)),
            "per_class": {label: per_class[label] for label in CLASS_ORDER},
            "confusion_matrix": {
                "labels": CLASS_ORDER,
                "matrix": matrix.astype(int).tolist(),
            },
        },
        "top_coefficients": top_coefficients(pipeline),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(report)
    print(f"Vocabulario: {summary['vectorizer']['vocabulary_size']}")
    print(f"F1-macro test: {summary['test']['f1_macro']:.4f}")
    print(f"Matriz guardada en {OUTPUT_DIR / 'matriz_confusion.png'}")


if __name__ == "__main__":
    main()
