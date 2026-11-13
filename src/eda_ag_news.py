"""EDA técnico del train de AG News. El test no entra en estas estadísticas."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from cargar_dataset import cargar  # noqa: E402
from src.preprocess import get_nlp, preprocess_corpus  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "eda"
CLASS_ORDER = ["World", "Sports", "Business", "Sci_Tech"]


def token_lengths(processed: pd.Series) -> np.ndarray:
    return processed.map(lambda text: 0 if not text else len(text.split())).to_numpy()


def content_stop_words() -> list[str]:
    """Une la lista de sklearn con las de SpaCy para cubrir lemmas como 'say'."""
    raw = set(ENGLISH_STOP_WORDS) | set(get_nlp().Defaults.stop_words)
    return sorted(word for word in raw if word.isalpha() and len(word) >= 2)


def top_ngrams(processed: pd.Series, n: int, top_k: int = 20) -> list[dict]:
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words=content_stop_words(),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    matrix = vectorizer.fit_transform(processed)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocabulary = np.array(vectorizer.get_feature_names_out())
    order = counts.argsort()[::-1][:top_k]
    return [
        {"ngram": vocabulary[index], "count": int(counts[index])}
        for index in order
    ]


def top_words(processed: pd.Series, stop_words: set[str], top_k: int = 50) -> list[dict]:
    vectorizer = CountVectorizer(
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    matrix = vectorizer.fit_transform(processed)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocabulary = np.array(vectorizer.get_feature_names_out())
    order = counts.argsort()[::-1][:top_k]
    rows = []
    for index in order:
        word = vocabulary[index]
        rows.append(
            {
                "word": word,
                "count": int(counts[index]),
                "is_stop": word in stop_words,
            }
        )
    return rows


def save_length_histogram(lengths: np.ndarray, percentile_95: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths, bins=40, color="#1f4e79", edgecolor="white")
    axis.axvline(
        percentile_95,
        color="#c45911",
        linestyle="--",
        linewidth=1.8,
        label=f"Percentil 95 = {percentile_95:.0f}",
    )
    axis.set_title("Tokens por noticia después del preprocesamiento")
    axis.set_xlabel("Cantidad de tokens")
    axis.set_ylabel("Cantidad de noticias")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_class_chart(counts: pd.Series, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels = [label for label in CLASS_ORDER if label in counts.index]
    values = [int(counts[label]) for label in labels]
    axis.bar(labels, values, color="#2e75b6")
    axis.set_ylim(0, max(values) * 1.15)
    axis.set_title("Documentos de entrenamiento por categoría")
    axis.set_xlabel("Clase")
    axis.set_ylabel("Cantidad de noticias")
    for position, value in enumerate(values):
        axis.text(position, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = cargar("ag_news")

    raw = train_df["text"].fillna("")
    html_docs = int(raw.str.contains(r"&lt;|&gt;|<[^>]+>", regex=True).sum())
    url_docs = int(raw.str.contains(r"https?://|www\.", regex=True).sum())

    processed = pd.Series(preprocess_corpus(raw.tolist()), name="text")
    lengths = token_lengths(processed)
    percentile_95 = float(np.percentile(lengths, 95))
    stop_words = set(get_nlp().Defaults.stop_words)
    words = top_words(processed, stop_words)
    stop_in_top = sum(row["is_stop"] for row in words)

    summary = {
        "dataset": "ag_news",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_used_in_eda": False,
        "classes": train_df["label"].value_counts().reindex(CLASS_ORDER).astype(int).to_dict(),
        "html_docs_in_train": html_docs,
        "url_docs_in_train": url_docs,
        "empty_after_preprocess": int((processed.str.len() == 0).sum()),
        "length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "mean": float(lengths.mean()),
            "p95": percentile_95,
            "max": int(lengths.max()),
            "recommended_max_len": int(np.ceil(percentile_95)),
        },
        "stop_words_in_top_50": int(stop_in_top),
        "top_50": words,
        "bigrams": top_ngrams(processed, 2),
        "trigrams": top_ngrams(processed, 3),
        "examples": [
            {"raw": raw.iloc[index], "processed": processed.iloc[index]}
            for index in (1, 10, 100)
        ],
    }

    save_length_histogram(lengths, percentile_95, OUTPUT_DIR / "longitud_tokens.png")
    save_class_chart(train_df["label"].value_counts(), OUTPUT_DIR / "distribucion_clases.png")
    pd.DataFrame(words).to_csv(OUTPUT_DIR / "top50.csv", index=False)
    pd.DataFrame(summary["bigrams"]).to_csv(OUTPUT_DIR / "bigramas.csv", index=False)
    pd.DataFrame(summary["trigrams"]).to_csv(OUTPUT_DIR / "trigramas.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "train_rows", "test_rows", "classes", "html_docs_in_train",
        "url_docs_in_train", "empty_after_preprocess", "length",
        "stop_words_in_top_50",
    )}, indent=2))
    print("BIGRAMS")
    for row in summary["bigrams"]:
        print(f"{row['count']:5d}  {row['ngram']}")
    print("TRIGRAMS")
    for row in summary["trigrams"]:
        print(f"{row['count']:5d}  {row['ngram']}")
    print("TOP50_STOP", stop_in_top)


if __name__ == "__main__":
    main()
