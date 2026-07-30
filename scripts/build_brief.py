"""Generate the polished two-page QUBOLens project brief."""

from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output" / "pdf" / "QUBOLens-Project-Brief.pdf"
SOCIAL_CARD = ROOT / "qubolens" / "web" / "og.png"

INK = HexColor("#0B0D0C")
INK_SOFT = HexColor("#171A18")
PAPER = HexColor("#F4F0E6")
PAPER_DEEP = HexColor("#E8E2D4")
ACID = HexColor("#D9FF70")
VIOLET = HexColor("#9F92FF")
CORAL = HexColor("#FF8D72")
MUTED = HexColor("#73766D")
WHITE = HexColor("#FFFDF7")

PAGE_W, PAGE_H = A4
MARGIN = 42


def paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y_top: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 10,
    leading: float = 14,
    color=PAPER,
    bold: bool = False,
) -> float:
    font_name = "Helvetica-Bold" if bold else font
    plain = re.sub(r"</?b>", "", text)
    lines: list[str] = []
    for segment in plain.split("<br/>"):
        lines.extend(simpleSplit(segment, font_name, size, width) or [""])
    text_object = pdf.beginText(x, y_top - size)
    text_object.setFont(font_name, size)
    text_object.setLeading(leading)
    text_object.setCharSpace(0)
    text_object.setFillColor(color)
    for rendered_line in lines:
        text_object.textLine(rendered_line)
    pdf.drawText(text_object)
    return y_top - len(lines) * leading


def label(pdf: canvas.Canvas, text: str, x: float, y: float, color=MUTED) -> None:
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", 6.8)
    pdf.drawString(x, y, text.upper())


def line(pdf: canvas.Canvas, x1: float, y: float, x2: float, color: Color) -> None:
    pdf.setStrokeColor(color)
    pdf.setLineWidth(0.6)
    pdf.line(x1, y, x2, y)


def rounded_card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: Color,
    stroke: Color | None = None,
    radius: float = 9,
) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke or fill)
    pdf.roundRect(x, y, width, height, radius, fill=1, stroke=1 if stroke else 0)


def draw_page_number(pdf: canvas.Canvas, page: int, dark: bool = False) -> None:
    color = Color(1, 1, 1, alpha=0.4) if dark else Color(0.05, 0.06, 0.05, alpha=0.45)
    pdf.setFillColor(color)
    pdf.setFont("Courier", 7)
    pdf.drawRightString(PAGE_W - MARGIN, 22, f"QUBOLENS / 0{page}")


def page_one(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(INK)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    image_height = 312
    image_width = PAGE_W
    if SOCIAL_CARD.exists():
        image = ImageReader(str(SOCIAL_CARD))
        pdf.drawImage(
            image,
            0,
            PAGE_H - image_height,
            width=image_width,
            height=image_height,
            preserveAspectRatio=False,
            mask="auto",
        )
    pdf.setFillColor(Color(0.043, 0.051, 0.047, alpha=0.22))
    pdf.rect(0, PAGE_H - image_height, PAGE_W, image_height, fill=1, stroke=0)

    label(pdf, "Open source engineering brief", MARGIN, PAGE_H - 28, ACID)
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawRightString(
        PAGE_W - MARGIN,
        PAGE_H - 28,
        "FEATURE SELECTION / VISUAL LAB",
    )

    y = PAGE_H - image_height - 32
    label(pdf, "The project", MARGIN, y, VIOLET)
    y -= 22
    title_bottom = paragraph(
        pdf,
        "A clearer way to choose<br/>which features deserve to stay.",
        MARGIN,
        y,
        420,
        size=25,
        leading=28,
        color=PAPER,
        bold=True,
    )
    y = title_bottom - 15
    paragraph(
        pdf,
        "QUBOLens is an open-source feature selection lab. Choose how many "
        "inputs to keep, then compare a smaller, less repetitive set with simple "
        "alternatives. Every result is visual, downloadable, and reproducible.",
        MARGIN,
        y,
        PAGE_W - 2 * MARGIN,
        size=10.2,
        leading=15,
        color=Color(0.956, 0.941, 0.902, alpha=0.72),
    )

    cards_y = 258
    gap = 8
    card_width = (PAGE_W - 2 * MARGIN - gap * 2) / 3
    facts = [
        ("2", "sample datasets"),
        ("40", "feature limit"),
        ("5x", "score comparison"),
    ]
    for index, (value, caption) in enumerate(facts):
        x = MARGIN + index * (card_width + gap)
        rounded_card(pdf, x, cards_y, card_width, 62, INK_SOFT, Color(1, 1, 1, alpha=0.1))
        pdf.setFillColor(ACID if index == 0 else PAPER)
        pdf.setFont("Courier-Bold", 20)
        pdf.drawString(x + 14, cards_y + 31, value)
        pdf.setFillColor(Color(1, 1, 1, alpha=0.42))
        pdf.setFont("Helvetica", 7.5)
        pdf.drawString(x + 14, cards_y + 14, caption)

    line(pdf, MARGIN, 233, PAGE_W - MARGIN, Color(1, 1, 1, alpha=0.12))
    label(pdf, "Why this is useful", MARGIN, 214, CORAL)

    column_gap = 40
    column_width = (PAGE_W - 2 * MARGIN - column_gap) / 2
    left_x = MARGIN
    right_x = MARGIN + column_width + column_gap
    paragraph(
        pdf,
        "<b>Every extra input has a cost.</b><br/>Features can require sensor "
        "reads, database joins, API calls, memory, and time. A simple ranking can "
        "keep two features that tell nearly the same story.",
        left_x,
        192,
        column_width,
        size=8.5,
        leading=12,
        color=Color(0.956, 0.941, 0.902, alpha=0.7),
    )
    paragraph(
        pdf,
        "<b>QUBOLens looks at features together.</b><br/>It keeps features that "
        "connect with the target, avoids repeated information, and always "
        "respects the limit you choose.",
        right_x,
        192,
        column_width,
        size=8.5,
        leading=12,
        color=Color(0.956, 0.941, 0.902, alpha=0.7),
    )

    rounded_card(pdf, MARGIN, 52, PAGE_W - 2 * MARGIN, 66, ACID)
    label(pdf, "Premise", MARGIN + 16, 99, INK)
    paragraph(
        pdf,
        "My takeaway: quantum ideas can be useful before quantum hardware is. "
        "The practical first step is <b>framing a messy choice as a clear "
        "optimization problem.</b>",
        MARGIN + 16,
        88,
        PAGE_W - 2 * MARGIN - 32,
        size=11.6,
        leading=15,
        color=INK,
    )
    draw_page_number(pdf, 1, dark=True)
    pdf.showPage()


def page_two(pdf: canvas.Canvas) -> None:
    pdf.setFillColor(PAPER)
    pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    label(pdf, "How it works", MARGIN, PAGE_H - 34, MUTED)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawString(MARGIN, PAGE_H - 68, "Useful now. Portable later.")

    formula_y = PAGE_H - 148
    rounded_card(pdf, MARGIN, formula_y, PAGE_W - 2 * MARGIN, 58, INK)
    pdf.setFillColor(ACID)
    pdf.setFont("Courier-Bold", 11)
    formula = "selection = useful signal + less repetition + your limit"
    pdf.drawCentredString(PAGE_W / 2, formula_y + 34, formula)
    pdf.setFillColor(Color(1, 1, 1, alpha=0.43))
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(
        PAGE_W / 2,
        formula_y + 16,
        "keep what helps / avoid what repeats / respect the feature limit",
    )

    top = formula_y - 27
    left_width = 248
    right_x = MARGIN + left_width + 32
    right_width = PAGE_W - MARGIN - right_x

    label(pdf, "Pipeline", MARGIN, top, VIOLET)
    steps = [
        (
            "01",
            "Measure what helps",
            "Each feature is checked against the target. Features are also "
            "compared with one another to find repeated information.",
        ),
        (
            "02",
            "Try many combinations",
            "A repeatable search explores different feature sets and gradually "
            "settles on the strongest option for the chosen limit.",
        ),
        (
            "03",
            "Compare fairly",
            "The chosen set is checked against a simple ranking and the full "
            "dataset, using the same score and data splits.",
        ),
    ]
    step_y = top - 27
    for number, title, body in steps:
        pdf.setFillColor(VIOLET if number != "03" else CORAL)
        pdf.setFont("Courier-Bold", 8)
        pdf.drawString(MARGIN, step_y, number)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(MARGIN + 31, step_y, title)
        body_bottom = paragraph(
            pdf,
            body,
            MARGIN + 31,
            step_y - 8,
            left_width - 31,
            size=8.5,
            leading=12,
            color=MUTED,
        )
        step_y = body_bottom - 19
        line(pdf, MARGIN + 31, step_y + 9, MARGIN + left_width, PAPER_DEEP)

    label(pdf, "What the user gets", right_x, top, CORAL)
    bullets = [
        "An exact feature limit and adjustable overlap control",
        "A score-versus-size chart and live search trace",
        "Chosen features and an interaction map",
        "A fair comparison with simple alternatives",
        "A downloadable report and technical matrix",
        "Web UI, Python API, CLI,<br/>tests, CI, and Render deploy",
    ]
    bullet_y = top - 24
    for item in bullets:
        pdf.setFillColor(ACID)
        pdf.circle(right_x + 3, bullet_y + 2, 2.2, fill=1, stroke=0)
        bullet_y = paragraph(
            pdf,
            item,
            right_x + 13,
            bullet_y + 7,
            right_width - 13,
            size=8.6,
            leading=11.5,
            color=INK,
        ) - 8

    card_y = 375
    rounded_card(pdf, right_x, card_y, right_width, 98, WHITE, PAPER_DEEP)
    label(pdf, "Credibility checks", right_x + 14, card_y + 78, INK)
    paragraph(
        pdf,
        "The math is tested state-by-state. Seeds control the data, search, and "
        "splits. Every method uses the same comparison. The interface clearly "
        "labels scores as exploratory.",
        right_x + 14,
        card_y + 65,
        right_width - 28,
        size=8.1,
        leading=11,
        color=MUTED,
    )

    line(pdf, MARGIN, 344, PAGE_W - MARGIN, PAPER_DEEP)
    label(pdf, "Boundaries are part of the product", MARGIN, 322, CORAL)
    paragraph(
        pdf,
        "<b>The included search is classical, not quantum computation.</b> "
        "Correlation misses some nonlinear interactions. Comparison scores are "
        "not a held-out production estimate. A lower search score does not "
        "guarantee a better downstream model. QUBOLens says all four plainly.",
        MARGIN,
        306,
        PAGE_W - 2 * MARGIN,
        size=9.2,
        leading=13,
        color=INK,
    )

    label(pdf, "Research basis", MARGIN, 224, VIOLET)
    references = [
        (
            "Mucke et al. (2023)",
            "Feature Selection on Quantum Computers - Quantum Machine Intelligence.",
        ),
        (
            "Glover, Kochenberger & Du (2018)",
            "A Tutorial on Formulating and Using QUBO Models.",
        ),
        (
            "Pranjic, Mummaneni & Tutschku (2024)",
            "Quantum Annealing based Feature Selection in Machine Learning.",
        ),
        (
            "Hellstern, Dehn & Zaefferer (2023)",
            "Quantum computer based Feature Selection in Machine Learning.",
        ),
    ]
    reference_y = 204
    for author, title in references:
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 7.4)
        pdf.drawString(MARGIN, reference_y, author)
        title_x = MARGIN + 225
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 7.4)
        available = PAGE_W - MARGIN - title_x
        rendered = title
        while stringWidth(rendered, "Helvetica", 7.4) > available:
            rendered = rendered[:-1]
        if rendered != title:
            rendered = rendered[:-3] + "..."
        pdf.drawString(title_x, reference_y, rendered)
        reference_y -= 18

    rounded_card(pdf, MARGIN, 55, PAGE_W - 2 * MARGIN, 72, INK)
    label(pdf, "Open source / MIT", MARGIN + 16, 106, ACID)
    pdf.setFillColor(PAPER)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN + 16, 83, "github.com/RS-010806/quantum")
    pdf.setFillColor(Color(1, 1, 1, alpha=0.45))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawRightString(
        PAGE_W - MARGIN - 16,
        83,
        "zero dependencies / one service",
    )
    pdf.linkURL(
        "https://github.com/RS-010806/quantum",
        (MARGIN + 12, 67, PAGE_W - MARGIN - 12, 118),
        relative=0,
    )
    draw_page_number(pdf, 2)
    pdf.showPage()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)
    pdf.setTitle("QUBOLens - Project Brief")
    pdf.setAuthor("Rishi")
    pdf.setSubject("A visual, open-source feature selection lab")
    page_one(pdf)
    page_two(pdf)
    pdf.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
