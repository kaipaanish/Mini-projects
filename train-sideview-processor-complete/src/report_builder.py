import os
from datetime import datetime
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def build_html_report(template_dir: str, out_html: str, context: Dict[str, Any]):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )
    tmpl = env.get_template("report.html.j2")
    html = tmpl.render(**context)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

def build_pdf_report(out_pdf: str, context: Dict[str, Any]):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    width, height = A4

    margin = 36
    y = height - margin

    def draw_text(text, size=14, dy=18):
        nonlocal y
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, y, text)
        y -= dy

    def draw_small(text, size=10, dy=14):
        nonlocal y
        c.setFont("Helvetica", size)
        c.drawString(margin, y, text)
        y -= dy

    draw_text("Train Side-View Report", 18, 22)
    draw_small(f"Train number: {context['train_number']}")
    draw_small(f"Coach count: {context['coach_count']}")
    draw_small(f"Input video: {context['input_video']}")
    draw_small(f"Generated: {context['generated_at']}")
    y -= 10

    for coach in context["coaches"]:
        if y < 200:
            c.showPage(); y = height - margin
        draw_text(f"Coach {coach['index']}  ({coach['video_name']})", 14, 18)

        # Images in a simple grid
        x = margin
        max_h = 0
        col_w = (width - 2*margin) / 3 - 6
        for i, fr in enumerate(coach["frames"]):
            try:
                img = ImageReader(fr["anno_path"])
                iw, ih = img.getSize()
                scale = min(col_w/iw, 160/ih)
                w = iw * scale
                h = ih * scale
                if x + w > width - margin:
                    x = margin
                    y -= (max_h + 20)
                    max_h = 0
                if y - h < margin:
                    c.showPage(); y = height - margin
                    draw_text(f"Coach {coach['index']}  ({coach['video_name']})", 14, 18)
                    x = margin; max_h = 0

                c.drawImage(img, x, y - h, width=w, height=h)
                # caption
                cap = fr["name"]
                if fr.get("detections"):
                    cap += " — " + ", ".join(fr["detections"])
                c.setFont("Helvetica", 8)
                c.drawString(x, y - h - 10, cap)
                x += (w + 12)
                max_h = max(max_h, h)
            except Exception:
                continue
        y -= (max_h + 28)

    c.save()
