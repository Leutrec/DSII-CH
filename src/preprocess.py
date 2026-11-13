"""Limpieza con regex, normalización y lematización para el corpus de noticias."""

import html
import re

import spacy

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_BROKEN_ENTITY = re.compile(r"(?:#\d+;|quot;|amp;)", re.IGNORECASE)
_TICKER_WIDGET = re.compile(
    r"\([^)]{0,40}:\s*quote,\s*profile,\s*research\)",
    re.IGNORECASE,
)
_WIRE = re.compile(r"\((?:reuters|afp|ap|canadian press)\)|\bcanadian press\b|\b(?:reuters|afp|ap)\b")
_DOMAIN = re.compile(r"\b[\w.-]+\.(?:com|net|org|info)\b")
_POSSESSIVE = re.compile(r"'s\b")
_NUMBER = re.compile(r"\b\d+(?:st|nd|rd|th)?\b")
_NON_LETTER = re.compile(r"[^a-z\s]")
_MULTISPACE = re.compile(r"\s+")

_NLP = None


def get_nlp():
    """Carga en_core_web_sm una sola vez. El tagger alcanza para lematizar."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def clean_text(text: str) -> str:
    """Saca HTML, URLs y signos, y deja el texto en minúsculas."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    text = text.replace("#39;", "'")
    text = _BROKEN_ENTITY.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _TICKER_WIDGET.sub(" ", text)
    text = text.lower()
    text = _POSSESSIVE.sub(" ", text)
    text = _NUMBER.sub(" ", text)
    text = _DOMAIN.sub(" ", text)
    text = _WIRE.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    return _MULTISPACE.sub(" ", text).strip()


def _lemmas_from_doc(doc) -> str:
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if token.is_alpha and lemma.isalpha() and lemma != "-pron-":
            lemmas.append(lemma)
    return " ".join(lemmas)


def preprocess_text(text: str) -> str:
    """Aplica limpieza y lematización. Devuelve lemmas separados por espacio."""
    return _lemmas_from_doc(get_nlp()(clean_text(text)))


def preprocess_corpus(texts) -> list[str]:
    """Equivalente a preprocess_text sobre una colección, usando nlp.pipe."""
    nlp = get_nlp()
    cleaned = [clean_text(text) for text in texts]
    return [_lemmas_from_doc(doc) for doc in nlp.pipe(cleaned, batch_size=128)]
