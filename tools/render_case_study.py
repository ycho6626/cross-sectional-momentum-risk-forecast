"""Render case_study.md as a one-page PDF with clickable Markdown links.

Run with optional documentation dependencies:
uv run --with reportlab --with pypdfium2 python tools/render_case_study.py . \
    examples/output/case_study_preview.png
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def markup(text: str) -> str:
    text = html.escape(text.replace("−", "-"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(
        r"\[([^\]]+)\]\((https://[^)]+)\)",
        r'<link href="\2" color="#007E87"><u>\1</u></link>', text,
    )


def render(root: Path, preview: Path) -> None:
    target = root / "docs/case_study.pdf"
    body = (root / "docs/case_study.md").read_text()
    navy = colors.HexColor("#173042")
    teal = colors.HexColor("#007E87")
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19,
                                leading=23, textColor=navy, spaceAfter=9),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9,
                                   leading=12, textColor=teal, spaceAfter=15),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10.3,
                                  leading=13, textColor=navy, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.8,
                               leading=13.1, textColor=colors.HexColor("#26343D"),
                               alignment=TA_LEFT, spaceAfter=6),
        "foot": ParagraphStyle("foot", fontName="Helvetica", fontSize=8,
                               leading=10.5, textColor=navy, spaceBefore=5),
    }
    blocks = body.strip().split("\n\n")
    story = []
    for index, block in enumerate(blocks):
        if block.startswith("# "):
            story.append(Paragraph(markup(block[2:]), styles["title"]))
        elif index == 1:
            story.append(Paragraph(markup(block), styles["subtitle"]))
        elif block.startswith("## "):
            story.append(Paragraph(markup(block[3:]), styles["section"]))
        elif block.startswith("**Repository:"):
            text = block.split("  \n")[0]
            story.append(Spacer(1, 5))
            story.append(Paragraph(markup(text), styles["foot"]))
            story.append(Paragraph("Exact statistics and validation receipts are linked in the "
                                   "repository. No licensed observations are distributed.",
                                   styles["foot"]))
        else:
            story.append(Paragraph(markup(block.replace("\n", " ")), styles["body"]))

    def frame(canvas, document):
        canvas.setTitle("Momenta | Quantitative Research Case Study")
        canvas.setAuthor("ycho6626")
        canvas.setSubject(
            "Momentum risk: mathematical motivation, research design and negative findings",
        )
        canvas.setCreator("Momenta Research")
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(2)
        canvas.line(45, A4[1] - 31, A4[0] - 45, A4[1] - 31)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(navy)
        canvas.drawString(45, 23, "RESEARCH PORTFOLIO   /   MOMENTA")
        canvas.drawRightString(A4[0] - 45, 23, str(document.page))

    document = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=45, leftMargin=45,
                                 topMargin=44, bottomMargin=39, invariant=1)
    document.build(story, onFirstPage=frame, onLaterPages=frame)
    pdf = pdfium.PdfDocument(target)
    if len(pdf) != 1:
        raise ValueError(f"Expected one page, obtained {len(pdf)}")
    preview.parent.mkdir(parents=True, exist_ok=True)
    pdf[0].render(scale=1.5).to_pil().save(preview)
    print(f"One-page PDF created: {target.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()
    render(args.root, args.preview)
