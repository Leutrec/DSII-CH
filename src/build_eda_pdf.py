"""Arma el PDF de la preentrega a partir de los resultados del EDA."""

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "outputs" / "eda" / "summary.json"
PDF_PATH = ROOT / "EDA_NLP_DiazLuis.pdf"
FONTS = Path(r"C:\Windows\Fonts")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


class Report(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"EDA_NLP_DiazLuis  |  {self.page_no()}", align="C")


def paragraph(pdf: Report, text: str, size: int = 11, style: str = "", height: float = 5.5) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Arial", style, size)
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


def section(pdf: Report, title: str) -> None:
    if pdf.get_y() > pdf.page_break_trigger - 30:
        pdf.add_page()
    pdf.ln(3)
    pdf.set_text_color(20, 20, 20)
    paragraph(pdf, title, size=13, style="B", height=7)
    pdf.ln(1)


def body(pdf: Report, text: str) -> None:
    pdf.set_text_color(30, 30, 30)
    paragraph(pdf, text)
    pdf.ln(1)


def bullet(pdf: Report, text: str) -> None:
    pdf.set_text_color(30, 30, 30)
    paragraph(pdf, f"- {text}")


def ngram_table(pdf: Report, rows: list[dict]) -> None:
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(31, 78, 121)
    pdf.set_text_color(255, 255, 255)
    pdf.set_x(pdf.l_margin)
    pdf.cell(14, 6, "#", border=0, fill=True, align="C")
    pdf.cell(126, 6, "N-grama", border=0, fill=True)
    pdf.cell(30, 6, "Frecuencia", border=0, fill=True, align="R")
    pdf.ln()
    for rank, row in enumerate(rows, start=1):
        pdf.set_x(pdf.l_margin)
        if rank % 2 == 0:
            pdf.set_fill_color(232, 239, 246)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Arial", "", 9)
        pdf.cell(14, 5, str(rank), fill=True, align="C")
        pdf.cell(126, 5, row["ngram"], fill=True)
        pdf.cell(30, 5, str(row["count"]), fill=True, align="R")
        pdf.ln()
    pdf.ln(2)


def conclusion_text(summary: dict) -> str:
    length = summary["length"]
    not_count = next(row["count"] for row in summary["top_50"] if row["word"] == "not")
    per_class = summary["classes"]["World"]
    return (
        f"El train de AG News reúne {thousands(summary['train_rows'])} noticias breves "
        f"y cuatro clases parejas, con {thousands(per_class)} documentos en World, Sports, "
        f"Business y Sci_Tech. El test suma {thousands(summary['test_rows'])} noticias y quedó "
        "fuera de este análisis, para no usarlo hasta la evaluación.\n\n"
        f"El texto de origen trae ruido de cable. Hay etiquetas HTML en "
        f"{summary['html_docs_in_train']} notas y una URL en {summary['url_docs_in_train']}. "
        "También aparecen entidades rotas (#39; y quot;), la firma de AP, Reuters o AFP "
        "repetida, y el accesorio de cotización Quote, Profile, Research. Eso no dice de qué "
        "trata la nota y, si se deja, llena los n-gramas.\n\n"
        "preprocess_text lo encadena. Repara esas entidades, borra etiquetas, URLs, dominios, "
        "la firma de agencia y el accesorio de cotización, pasa a minúsculas, quita puntuación "
        "y números, y lematiza con SpaCy (en_core_web_sm). Ninguna noticia quedó vacía. "
        f"La mediana es de {length['median']:.0f} tokens y el percentil 95 es "
        f"{length['p95']:.0f}; el máximo llega a {length['max']}. Conviene usar "
        f"{length['recommended_max_len']} como max_len: cubre el 95 % de las notas y no "
        "guarda lugar para la cola larga.\n\n"
        f"De las 50 palabras más frecuentes, {summary['stop_words_in_top_50']} son stop-words. "
        "Dominan the, to, of, be y say, este último lema de said. No las saqué del corpus. "
        f"not figura {not_count} veces y puede dar vuelta el sentido. En el módulo 3, TF-IDF "
        "les baja el peso. Las excluí solo al ordenar los n-gramas, para que la tabla muestre el tema.\n\n"
        "Esos bi-gramas y tri-gramas ya son frases de noticia, como oil price, prime minister, "
        "new york stock e international space station. No aparece HTML ni un dominio. "
        "world large es el lema de world's largest, no una marca. Como las clases están "
        "equilibradas, no hace falta remuestrear. Lo que el modelo tiene que tolerar es esta "
        f"prosa de agencia, muy formulaica, y el recorte a {length['recommended_max_len']} tokens."
    )


def write_report(summary: dict) -> None:
    conclusion = conclusion_text(summary)
    word_count = len(conclusion.replace("\n", " ").split())
    if word_count > 300:
        raise SystemExit(f"La conclusión tiene {word_count} palabras; el máximo es 300.")

    pdf = Report(format="A4")
    pdf.add_font("Arial", "", str(FONTS / "arial.ttf"))
    pdf.add_font("Arial", "B", str(FONTS / "arialbd.ttf"))
    pdf.add_font("Arial", "I", str(FONTS / "ariali.ttf"))
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    pdf.set_text_color(20, 20, 20)
    paragraph(pdf, "EDA técnico del corpus AG News", size=18, style="B", height=8)
    pdf.set_text_color(60, 60, 60)
    paragraph(pdf, "Preentrega del Módulo 2. Pipeline de preprocesamiento y diagnóstico.", height=6)
    paragraph(pdf, "Luis Diaz  |  Data Science III", height=6)
    pdf.ln(2)

    section(pdf, "1. Corpus y splits")
    body(
        pdf,
        "Se usó el dataset provisto por la cátedra, cargado con cargar_dataset.py. "
        "La elección queda fija para el resto del programa: AG News, noticias en inglés "
        "con cuatro categorías. Todo el diagnóstico se calculó sobre ag_news_train.csv. "
        f"ag_news_test.csv ({thousands(summary['test_rows'])} filas) no se preprocesó ni "
        "se contó en estas tablas.",
    )
    per_class = summary["classes"]
    body(
        pdf,
        "Train: "
        + ", ".join(f"{name} {thousands(per_class[name])}" for name in per_class)
        + f". Total: {thousands(summary['train_rows'])} documentos. Columnas: text y label.",
    )

    section(pdf, "2. Pasos de preprocesamiento")
    body(pdf, "La función preprocess_text aplica, en este orden:")
    for step in (
        "Reparar entidades rotas del export (#39; como apóstrofe, quot; y amp;) y decodificar el HTML que sí trae ampersand.",
        "Borrar etiquetas HTML, URLs, dominios .com/.net/.org y el widget (Ticker: Quote, Profile, Research).",
        "Sacar la firma de agencia (AP, Reuters, AFP, Canadian Press), que se repite en el titular y en el cuerpo.",
        "Pasar a minúsculas, eliminar posesivos 's, números y puntuación.",
        "Lematizar con el modelo en_core_web_sm de SpaCy y conservar solo tokens alfabéticos.",
    ):
        bullet(pdf, step)
    pdf.ln(1)

    example = next(item for item in summary["examples"] if "#39;" in item["raw"])
    body(pdf, "Ejemplo real del train, antes y después:")
    pdf.set_text_color(40, 40, 40)
    paragraph(pdf, "Original: " + example["raw"], size=9, style="I", height=4.5)
    pdf.ln(1)
    paragraph(pdf, "Limpio: " + example["processed"], size=9, style="I", height=4.5)
    pdf.ln(2)

    section(pdf, "3. Longitud de tokens")
    length = summary["length"]
    body(
        pdf,
        "Cada documento se contó por la cantidad de lemmas que dejó el pipeline. "
        f"Mínimo {length['min']}, mediana {length['median']:.0f}, media {length['mean']:.1f}, "
        f"percentil 95 igual a {length['p95']:.0f} y máximo {length['max']}. "
        f"El max_len recomendado para los módulos siguientes es {length['recommended_max_len']}.",
    )
    pdf.image(str(ROOT / "outputs" / "eda" / "longitud_tokens.png"), w=170)
    pdf.ln(2)

    section(pdf, "4. Distribución de clases")
    body(
        pdf,
        "Las cuatro categorías tienen la misma cantidad de noticias en el train. "
        "No hay clase mayoritaria. En los módulos 3 y 4 no hace falta balancear el muestreo. "
        "Conviene igual mirar F1 macro, porque no se esconde detrás del tamaño de una clase.",
    )
    pdf.image(str(ROOT / "outputs" / "eda" / "distribucion_clases.png"), w=170)
    pdf.ln(2)

    section(pdf, "5. Veinte bi-gramas más frecuentes")
    body(
        pdf,
        "Se obtuvieron con CountVectorizer de scikit-learn sobre el texto ya lematizado. "
        "Las stop-words se excluyeron solo en este ranking. Las frases que quedan son de tema: "
        "lugares, cargos, mercados y deporte. oil price y crude oil anticipan la clase Business; "
        "red sox y world cup, Sports; prime minister y united states, World. "
        "world large no es residuo de HTML: es el lema de world's largest, porque SpaCy reduce largest a large.",
    )
    ngram_table(pdf, summary["bigrams"])

    section(pdf, "6. Veinte tri-gramas más frecuentes")
    body(
        pdf,
        "Los tri-gramas confirman que la limpieza sacó el marcado. No hay href, http ni etiquetas. "
        "Aparecen nombres y hechos: prime minister ariel sharon, president george bush, "
        "international space station, security exchange commission, boston red sox. "
        "new york stock concentra el lenguaje de mercados. Esa repetición formulaica es propia "
        "del cable de agencia y sirve para clasificar por tema.",
    )
    ngram_table(pdf, summary["trigrams"])

    section(pdf, "7. Cincuenta palabras más frecuentes")
    content = [row["word"] for row in summary["top_50"] if not row["is_stop"]]
    body(
        pdf,
        f"{summary['stop_words_in_top_50']} de las 50 formas más frecuentes son stop-words. "
        "La cabeza es the, to, of, in, be, and. Entre las que no lo son aparecen "
        + ", ".join(content)
        + ". say es stop-word y es el lema de said, el verbo típico del cable. "
        "Los días de la semana también pesan, porque casi toda nota está fechada. "
        "Dejar estas palabras en el texto no impide modelar: TF-IDF les asigna poco peso. "
        "Borrarlas todas sacaría not, que sigue en el top 50 y cambia la polaridad de una frase.",
    )

    section(pdf, "8. Conclusión")
    for block in conclusion.split("\n\n"):
        body(pdf, block)

    pdf.output(str(PDF_PATH))
    print(f"PDF: {PDF_PATH}")
    print(f"Conclusión: {word_count} palabras")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    write_report(summary)


if __name__ == "__main__":
    main()
