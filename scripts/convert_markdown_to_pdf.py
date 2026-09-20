"""
scripts/convert_markdown_to_pdf.py
----------------------------------
Converts RF showcase Markdown reports into publication-grade PDFs using
Chrome Headless and high-DPI image embedding.

Outputs:
  - outputs/ast_classification_rf_visualizations.pdf
  - outputs/classification_and_rf_showcase.pdf
  - outputs/classification/astonishing_rf_visualizations.pdf
  - outputs/classification/classification_and_rf_showcase.pdf
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
CLASSIFICATION_DIR = OUTPUTS_DIR / "classification"


def image_to_base64(img_path: Path) -> str:
    """Read image file and return base64 data URI."""
    ext = img_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{data}"


def markdown_to_html(md_text: str, base_dir: Path) -> str:
    """Convert showcase markdown to styled HTML with embedded base64 images."""
    lines = md_text.splitlines()
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Handle list items
        if stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            item_text = stripped[2:]
            # Format bold and inline math
            item_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", item_text)
            item_text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", item_text)
            item_text = re.sub(r"`(.*?)`", r"<code>\1</code>", item_text)
            item_text = re.sub(r"\$(.*?)\$", r"<em>\1</em>", item_text)
            html_parts.append(f"  <li>{item_text}</li>")
            continue
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False

        if not stripped:
            continue

        # Handle Markdown images: ![caption](image_path)
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img_match:
            caption = img_match.group(1)
            raw_path = img_match.group(2)
            
            # Resolve image path
            img_file = Path(raw_path)
            if not img_file.is_absolute():
                img_file = (base_dir / raw_path).resolve()

            if not img_file.exists():
                # Try in outputs/figures/
                alt_file = OUTPUTS_DIR / "figures" / img_file.name
                if alt_file.exists():
                    img_file = alt_file

            if img_file.exists():
                b64_uri = image_to_base64(img_file)
                html_parts.append(
                    f'<figure class="showcase-figure">\n'
                    f'  <img src="{b64_uri}" alt="{caption}" />\n'
                    f'  <figcaption>{caption}</figcaption>\n'
                    f'</figure>'
                )
            else:
                html_parts.append(f'<p class="error-msg">[Missing Image: {img_file.name}]</p>')
            continue

        # Headers
        if stripped.startswith("# "):
            title = stripped[2:]
            html_parts.append(f'<h1 class="main-title">{title}</h1>')
        elif stripped.startswith("## "):
            sec_title = stripped[3:]
            html_parts.append(f'<div class="section-divider"></div>')
            html_parts.append(f'<h2 class="section-title">{sec_title}</h2>')
        elif stripped.startswith("### "):
            sub_title = stripped[4:]
            html_parts.append(f'<h3 class="subsection-title">{sub_title}</h3>')
        elif stripped == "---":
            html_parts.append('<hr class="divider" />')
        else:
            # Paragraph formatting
            p_text = stripped
            p_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", p_text)
            p_text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", p_text)
            p_text = re.sub(r"`(.*?)`", r"<code>\1</code>", p_text)
            p_text = re.sub(r"\$(.*?)\$", r"<em>\1</em>", p_text)
            html_parts.append(f'<p class="body-text">{p_text}</p>')

    if in_list:
        html_parts.append("</ul>")

    body_content = "\n".join(html_parts)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RF Analysis Showcase</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 12mm 14mm 12mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}

  * {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #0b0c10;
    color: #e0e0e8;
    line-height: 1.5;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
  }}

  .main-title {{
    font-size: 17pt;
    font-weight: 800;
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff44;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 8px;
    letter-spacing: -0.3px;
  }}

  .section-divider {{
    page-break-before: auto;
  }}

  .section-title {{
    font-size: 12.5pt;
    font-weight: 700;
    color: #ffffff;
    background: linear-gradient(90deg, #181926 0%, #0b0c10 100%);
    padding: 6px 10px;
    border-left: 4px solid #76ff03;
    border-radius: 2px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  .subsection-title {{
    font-size: 10.5pt;
    font-weight: 600;
    color: #ffd54f;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  .body-text {{
    color: #cccccc;
    margin: 6px 0;
    font-size: 9.5pt;
  }}

  ul {{
    margin: 6px 0 12px 18px;
    padding: 0;
  }}

  li {{
    color: #bbbbcc;
    margin-bottom: 4px;
    font-size: 9.0pt;
  }}

  strong {{
    color: #ffffff;
  }}

  code {{
    background: #1e1f29;
    color: #00e5ff;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.5pt;
  }}

  .divider {{
    border: none;
    border-top: 1px solid #252633;
    margin: 16px 0;
  }}

  .showcase-figure {{
    margin: 10px 0 16px 0;
    padding: 0;
    text-align: center;
    page-break-inside: avoid;
  }}

  .showcase-figure img {{
    width: 100%;
    max-height: 225mm;
    object-fit: contain;
    border: 1px solid #2c2d3d;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
    background-color: #050508;
    display: block;
    margin: 0 auto;
  }}

  figcaption {{
    font-size: 8pt;
    color: #888899;
    font-style: italic;
    margin-top: 5px;
    text-align: center;
    page-break-before: avoid;
  }}

  .error-msg {{
    color: #ff5252;
    background: #2a1115;
    padding: 6px;
    border-radius: 4px;
    font-weight: bold;
  }}
</style>
</head>
<body>
{body_content}
</body>
</html>
"""
    return html_template


def convert_file(md_path: Path, output_pdf_path: Path):
    """Convert a single markdown file to PDF using Chrome Headless."""
    print(f"Converting: {md_path.name} -> {output_pdf_path.name}")
    if not md_path.exists():
        print(f"  Error: {md_path} does not exist.")
        return False

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = markdown_to_html(md_text, base_dir=md_path.parent)

    tmp_html = md_path.parent / f"_temp_{md_path.stem}.html"
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer",
        str(tmp_html),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if output_pdf_path.exists() and output_pdf_path.stat().st_size > 0:
            size_kb = output_pdf_path.stat().st_size / 1024
            print(f"  Success: {output_pdf_path.name} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  Error: PDF was not generated or is empty.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  Chrome execution failed: {e.stderr}")
        return False
    finally:
        if tmp_html.exists():
            tmp_html.unlink()


def main():
    print("=" * 65)
    print("Converting RF Showcase Markdown Reports to Publication-Grade PDFs")
    print("=" * 65)

    if not Path(CHROME_PATH).exists():
        print(f"Error: Chrome not found at {CHROME_PATH}")
        sys.exit(1)

    tasks = [
        (
            OUTPUTS_DIR / "unified_rf_pipeline_showcase.md",
            OUTPUTS_DIR / "unified_rf_pipeline_showcase.pdf",
        ),
        (
            OUTPUTS_DIR / "ast_classification_rf_visualizations.md",
            OUTPUTS_DIR / "ast_classification_rf_visualizations.pdf",
        ),
        (
            OUTPUTS_DIR / "classification_and_rf_showcase.md",
            OUTPUTS_DIR / "classification_and_rf_showcase.pdf",
        ),
        (
            OUTPUTS_DIR / "rml_classification_rf_visualizations.md",
            OUTPUTS_DIR / "rml_classification_rf_visualizations.pdf",
        ),
        (
            CLASSIFICATION_DIR / "astonishing_rf_visualizations.md",
            CLASSIFICATION_DIR / "astonishing_rf_visualizations.pdf",
        ),
        (
            CLASSIFICATION_DIR / "classification_and_rf_showcase.md",
            CLASSIFICATION_DIR / "classification_and_rf_showcase.pdf",
        ),
    ]

    success_count = 0
    for md_p, pdf_p in tasks:
        if md_p.exists():
            ok = convert_file(md_p, pdf_p)
            if ok:
                success_count += 1

    print("=" * 65)
    print(f"Successfully generated {success_count} PDF reports.")
    print("=" * 65)


if __name__ == "__main__":
    main()
