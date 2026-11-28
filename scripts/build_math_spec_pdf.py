"""Generate clean, print-ready PDF from MATHEMATICAL_SPECIFICATION.md via 2-pass Playwright Chromium + pypdf TOC sync."""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "MATHEMATICAL_SPECIFICATION.md"
OUT_PDF = REPO_ROOT / "docs" / "Momentum_Strategy_Mathematical_Specification.pdf"

LANDSCAPE_SECTIONS = {
    "7.1 Clean ETF-Free 16-Year Zero-Survivorship PIT",
    "7.2 Live Synchronized Current-Constituent Panel",
    "9. Explicit Parameter Provenance & System Boundaries",
}


def _md_inline(text: str) -> str:
    """Convert inline markdown (**bold**, *em*, `code`, links, and $...$ math) cleanly to HTML."""
    math_spans: list[str] = []
    code_spans: list[str] = []

    def _save_math(m: re.Match) -> str:
        math_spans.append(m.group(1))
        return f"@@IMATH_{len(math_spans) - 1}@@"

    def _save_code(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return f"@@ICODE_{len(code_spans) - 1}@@"

    # Protect math and inline code BEFORE escaping and bold/italic parsing
    # (prevents asterisks inside code spans like `[0.25]*4` from breaking surrounding **...**)
    s = re.sub(r"\$([^$\n]+?)\$", _save_math, text)
    s = re.sub(r"`([^`\n]+)`", _save_code, s)
    s = html.escape(s)

    # Links, bold, italic
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)

    for idx, code in enumerate(code_spans):
        s = s.replace(f"@@ICODE_{idx}@@", f"<code>{html.escape(code)}</code>")
    for idx, tex in enumerate(math_spans):
        s = s.replace(f"@@IMATH_{idx}@@", f'<span class="math-inline">\\({html.escape(tex)}\\)</span>')
    return s


def _parse_markdown_blocks(md_text: str) -> tuple[str, list[dict]]:
    """Parse markdown into ordered HTML sections & metadata for TOC construction."""
    lines = md_text.splitlines()
    blocks: list[dict] = []
    title = "Mathematical & Systems Specification"
    i = 0

    def _flush_table(tbl_lines: list[str]) -> str:
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in tbl_lines]
        if len(rows) < 2:
            return ""
        headers = rows[0]
        aligns = []
        for sep in rows[1]:
            if sep.startswith(":") and sep.endswith(":"):
                aligns.append("center")
            elif sep.endswith(":"):
                aligns.append("right")
            else:
                aligns.append("left")
        body = rows[2:]
        if len(headers) == 10:
            cols_w = [30.5, 10.5, 7.2, 7.6, 6.2, 6.2, 6.5, 6.8, 8.0, 10.5]
        elif len(headers) >= 8:
            rem_w = 70.0 / (len(headers) - 1)
            cols_w = [30.0] + [rem_w] * (len(headers) - 1)
        elif len(headers) == 4 and "Parameter" in headers[0]:
            cols_w = [16.0, 16.0, 40.0, 28.0]
        elif len(headers) == 5 and "Layer" in headers[0]:
            cols_w = [21.0, 24.0, 14.0, 31.0, 10.0]
        else:
            cols_w = [100.0 / len(headers)] * len(headers)
        colgroup = "<colgroup>" + "".join(f'<col style="width:{w:.2f}%">' for w in cols_w) + "</colgroup>"
        th_html = "".join(
            f'<th style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{_md_inline(h)}</th>'
            for idx, h in enumerate(headers)
        )
        tr_html = []
        for r in body:
            tds = "".join(
                f'<td style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{_md_inline(c)}</td>'
                for idx, c in enumerate(r)
            )
            tr_html.append(f"<tr>{tds}</tr>")
        return f'<table>{colgroup}<thead><tr>{th_html}</tr></thead><tbody>{"".join(tr_html)}</tbody></table>'

    cur_sec_title = "Overview"
    cur_html: list[str] = []
    is_landscape = False
    is_h2 = False

    def _push_section():
        nonlocal cur_html, cur_sec_title, is_landscape, is_h2
        if cur_html:
            blocks.append({
                "title": cur_sec_title,
                "html": "\n".join(cur_html),
                "landscape": is_landscape,
                "is_h2": is_h2,
            })
        cur_html = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("# ") and i < 5:
            title = s[2:].strip()
            i += 1
            continue
        if s == "---":
            i += 1
            continue
        if s.startswith("## "):
            _push_section()
            cur_sec_title = s[3:].strip()
            is_h2 = True
            # Peek ahead: if only blank lines separate this H2 from a landscape H3, start landscape here
            want_land = any(cur_sec_title.startswith(p) for p in LANDSCAPE_SECTIONS)
            if not want_land:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].strip().startswith("### "):
                    next_h3 = lines[j].strip()[4:].strip()
                    want_land = any(next_h3.startswith(p) for p in LANDSCAPE_SECTIONS)
            is_landscape = want_land
            sec_id = f"sec_{len(blocks)}"
            cur_html.append(f'<h2 id="{sec_id}">{_md_inline(cur_sec_title)}</h2>')
            i += 1
            continue
        if s.startswith("### "):
            sub_t = s[4:].strip()
            want_land = any(sub_t.startswith(p) for p in LANDSCAPE_SECTIONS)
            if want_land != is_landscape:
                _push_section()
                cur_sec_title = sub_t
                is_landscape = want_land
                is_h2 = False
            cur_html.append(f"<h3>{_md_inline(sub_t)}</h3>")
            i += 1
            continue
        if s.startswith("$$"):
            if s.endswith("$$") and len(s) > 4:
                expr = s[2:-2].strip()
                cur_html.append(f'<div class="math-block">\\[{html.escape(expr)}\\]</div>')
                i += 1
                continue
            buf = [s[2:]]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                buf.append(lines[i])
                i += 1
            if i < len(lines):
                buf.append(lines[i].strip()[:-2])
                i += 1
            expr = "\n".join(buf).strip()
            cur_html.append(f'<div class="math-block">\\[{html.escape(expr)}\\]</div>')
            continue
        if s.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|") and "-" in lines[i + 1]:
            tbl = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            cur_html.append(_flush_table(tbl))
            continue
        if s.startswith("> "):
            bq = [s[2:]]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("> "):
                bq.append(lines[i].strip()[2:])
                i += 1
            cur_html.append(f'<blockquote>{_md_inline(" ".join(bq))}</blockquote>')
            continue
        if re.match(r"^(\d+\.|-)\s+", s):
            base_indent = len(ln) - len(ln.lstrip(" "))
            ord_m = re.match(r"^(\d+)\.\s+", s)
            is_ord = bool(ord_m)
            start_num = int(ord_m.group(1)) if ord_m else 1
            tag = "ol" if is_ord else "ul"
            items = []
            while i < len(lines):
                raw_ln = lines[i]
                cur_s = raw_ln.strip()
                cur_indent = len(raw_ln) - len(raw_ln.lstrip(" "))
                if not cur_s:
                    i += 1
                    break
                # Top-level list item at same indentation level
                if cur_indent <= base_indent + 1 and re.match(r"^(\d+\.|-)\s+", cur_s):
                    # Break if list type switched at top level
                    item_ord = bool(re.match(r"^\d+\.\s+", cur_s))
                    if item_ord != is_ord:
                        break
                    item_txt = re.sub(r"^(\d+\.|-)\s+", "", cur_s)
                    i += 1
                    while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*(\d+\.|-|\$\$|\|)", lines[i]):
                        item_txt += " " + lines[i].strip()
                        i += 1
                    items.append(f"<li>{_md_inline(item_txt)}</li>")
                # Nested bullet list inside parent list item
                elif cur_indent >= base_indent + 2 and items and cur_s.startswith("- "):
                    sub_bullets = []
                    while i < len(lines):
                        sl = lines[i]
                        ss = sl.strip()
                        si = len(sl) - len(sl.lstrip(" "))
                        if not ss or si < base_indent + 2 or not ss.startswith("- "):
                            break
                        sub_txt = ss[2:].strip()
                        i += 1
                        while i < len(lines) and (len(lines[i]) - len(lines[i].lstrip(" "))) > si and not lines[i].strip().startswith(("- ", "$$", "|")):
                            sub_txt += " " + lines[i].strip()
                            i += 1
                        sub_bullets.append(f"<li>{_md_inline(sub_txt)}</li>")
                    items[-1] = items[-1][:-5] + f'<ul class="sub-ul">{"".join(sub_bullets)}</ul></li>'
                elif cur_indent >= base_indent + 2 and items and cur_s.startswith("$$"):
                    expr_ln = cur_s
                    if expr_ln.endswith("$$") and len(expr_ln) > 4:
                        expr = expr_ln[2:-2].strip()
                        items[-1] = items[-1][:-5] + f'<div class="math-block">\\[{html.escape(expr)}\\]</div></li>'
                        i += 1
                    else:
                        break
                elif cur_indent >= base_indent + 2 and items:
                    items[-1] = items[-1][:-5] + f'<div class="sub-li">{_md_inline(cur_s)}</div></li>'
                    i += 1
                else:
                    break
            open_tag = f'<ol start="{start_num}">' if is_ord else "<ul>"
            cur_html.append(f"{open_tag}{''.join(items)}</{tag}>")
            continue
        if s:
            para = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|---|\||>|\$\$|\d+\.\s|-\s)", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            joined_p = " ".join(para)
            cls_attr = ' class="math-label"' if joined_p.endswith(":") else ""
            cur_html.append(f"<p{cls_attr}>{_md_inline(joined_p)}</p>")
            continue
        i += 1

    _push_section()
    return title, blocks


def _build_html_doc(title: str, blocks: list[dict], page_map: dict[str, str] | None = None) -> str:
    toc_rows = []
    for idx, b in enumerate(blocks):
        if b["title"] == "Overview" or not b.get("is_h2", True):
            continue
        pg_lbl = page_map.get(b["title"], f"{idx + 2:02d}") if page_map else f"{idx + 2:02d}"
        toc_rows.append(
            f'<tr><td class="toc-title">{_md_inline(b["title"])}</td>'
            f'<td class="toc-pg">{html.escape(pg_lbl)}</td></tr>'
        )

    body_sections = []
    for b in blocks:
        cls = "landscape-sec" if b["landscape"] else "portrait-sec"
        body_sections.append(f'<section class="{cls}">{b["html"]}</section>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  @page {{
    size: A4 portrait;
    margin: 16mm 14mm 18mm 14mm;
    @bottom-left {{
      content: "DIY Nifty Midcap 150 Momentum (R3) · Mathematical Specification";
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 8.5pt;
      color: #64748b;
    }}
    @bottom-right {{
      content: "Page " counter(page) " of " counter(pages);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 8.5pt;
      color: #64748b;
    }}
  }}
  @page landscape {{
    size: A4 landscape;
    margin: 13mm 12mm 16mm 12mm;
    @bottom-left {{
      content: "DIY Nifty Midcap 150 Momentum (R3) · Empirical Audit & Provenance";
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 8.5pt;
      color: #64748b;
    }}
    @bottom-right {{
      content: "Page " counter(page) " of " counter(pages);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 8.5pt;
      color: #64748b;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 9.8pt;
    line-height: 1.5;
    color: #0f172a;
    margin: 0;
  }}
  .cover-page {{
    break-after: page;
    padding-top: 26mm;
  }}
  .cover-kicker {{
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #059669;
    margin-bottom: 10px;
  }}
  .cover-title {{
    font-size: 22pt;
    font-weight: 700;
    line-height: 1.2;
    color: #0f172a;
    margin: 0 0 12px;
  }}
  .cover-sub {{
    font-size: 10.5pt;
    color: #475569;
    margin: 0 0 22px;
  }}
  .disclaimer-box {{
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 8.8pt;
    color: #334155;
    margin-bottom: 22px;
  }}
  .toc-heading {{
    font-size: 12pt;
    font-weight: 700;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #0f172a;
    padding-bottom: 4px;
  }}
  .toc-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
  }}
  .toc-table td {{
    padding: 5px 4px;
    border-bottom: 1px dashed #cbd5e1;
  }}
  .toc-pg {{
    text-align: right;
    font-family: "SFMono-Regular", Menlo, monospace;
    font-weight: 600;
    width: 56px;
  }}
  .portrait-sec {{ page: auto; }}
  .landscape-sec {{
    page: landscape;
    break-before: page;
    break-after: page;
  }}
  h2 {{
    font-size: 13.5pt;
    color: #0f172a;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin: 18px 0 10px;
    break-after: avoid;
  }}
  h3 {{
    font-size: 11pt;
    color: #1e293b;
    margin: 14px 0 6px;
    break-after: avoid;
  }}
  p {{ margin: 6px 0 8px; }}
  p.math-label {{ break-after: avoid; }}
  ul, ol {{ margin: 6px 0 10px 18px; padding: 0; }}
  li {{ margin: 4px 0; }}
  .sub-li {{ margin: 3px 0 3px 6px; color: #334155; }}
  .sub-ul {{ margin: 4px 0 4px 18px; }}
  .cover-title code {{ font-size: inherit; background: #f1f5f9; padding: 2px 6px; }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 8.4pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
  }}
  blockquote {{
    margin: 8px 0;
    padding: 8px 12px;
    background: #f8fafc;
    border-left: 3px solid #059669;
    font-size: 9.2pt;
    color: #334155;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 14px;
    font-size: 8.0pt;
    table-layout: fixed;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }}
  .landscape-sec table {{ font-size: 7.4pt; }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 4.5px 5.5px;
    vertical-align: top;
  }}
  .landscape-sec th, .landscape-sec td {{
    padding: 3.5px 4.5px;
  }}
  .landscape-sec td[style*="right"] {{
    white-space: nowrap;
  }}
  th {{
    background: #f1f5f9;
    font-weight: 700;
    color: #0f172a;
  }}
  tr {{ break-inside: avoid; }}
  .math-block {{
    margin: 8px 0 10px;
    text-align: center;
    overflow-x: visible;
    break-inside: avoid;
  }}
  /* Suppress duplicate hidden MathML layer so pypdf text extraction never scrambles glyphs */
  .katex-mathml {{ display: none !important; }}
  .katex {{ font-size: 1.0em; }}
  a {{ color: #047857; text-decoration: none; }}
</style>
</head>
<body>
  <div class="cover-page">
    <div class="cover-kicker">HarjasChadha / midcap-momentum-index · V3 / R3 Audit Specification</div>
    <h1 class="cover-title">{_md_inline(title)}</h1>
    <p class="cover-sub">Canonical Production &amp; Research Formulas, Net-of-Sec-50AA FIFO Backtest Verification, Point-in-Time NSE Universe Construction, and Autonomous Paper-Trading Ledger Architecture.</p>
    <div class="disclaimer-box">
      <strong>Disclosure &amp; Scope:</strong> Personal Quantitative Research &amp; Paper-Trading Sandbox — Not Investment Advice / Not SEBI Registered. All metrics reflect modeled Indian delivery/tax rules, 15 bps scheduled / 35 bps risk-off slippage, and Sec 50AA 30% liquid/debt taxation unless explicitly labeled as a comparison column.
    </div>
    <div class="toc-heading">Table of Contents</div>
    <table class="toc-table">
      <tbody>{"".join(toc_rows)}</tbody>
    </table>
  </div>
  {"".join(body_sections)}
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      if (window.renderMathInElement) {{
        window.renderMathInElement(document.body, {{
          delimiters: [
            {{left: '\\\\[', right: '\\\\]', display: true}},
            {{left: '\\\\(', right: '\\\\)', display: false}},
          ],
          throwOnError: false,
          output: 'html'
        }});
      }}
      window.__MATH_READY = true;
    }});
  </script>
</body>
</html>"""


def _render_pdf(html_content: str, pdf_path: Path, png_dir: Path | None = None) -> list[str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 1600})
        page.set_content(html_content, wait_until="networkidle", timeout=45000)
        page.wait_for_function("() => window.__MATH_READY === true", timeout=20000)
        page.evaluate("() => document.fonts ? document.fonts.ready : true")
        page.wait_for_timeout(400)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        page.pdf(path=str(pdf_path), print_background=True, prefer_css_page_size=True)
        if png_dir:
            png_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(png_dir / "html_preview_top.png"))
        browser.close()

    reader = PdfReader(str(pdf_path))
    return [pg.extract_text() or "" for pg in reader.pages]


def build_pdf() -> Path:
    md_text = MD_PATH.read_text()
    title, blocks = _parse_markdown_blocks(md_text)

    with tempfile.TemporaryDirectory() as tmpd:
        pass1_pdf = Path(tmpd) / "pass1.pdf"
        html1 = _build_html_doc(title, blocks, page_map=None)
        pages1 = _render_pdf(html1, pass1_pdf)

    # Resolve exact 1-indexed page numbers (skip cover page idx=0 & synthetic Overview / non-H2 wrappers)
    page_map: dict[str, str] = {}
    for b in blocks:
        if b["title"] == "Overview" or not b.get("is_h2", True):
            continue
        raw_t = re.sub(r"`|\*\*|\*|\[([^\]]+)\]\([^)]+\)", lambda m: m.group(1) if m.groups() and m.group(1) else "", b["title"])
        # Strip inline tex for plain text match
        prefix = re.sub(r"\$[^$]+\$", "", raw_t).split("(", 1)[0].strip()
        found_pg = None
        for idx in range(1, len(pages1)):
            if prefix in " ".join(pages1[idx].split()):
                found_pg = idx + 1
                break
        if found_pg is None:
            raise RuntimeError(f"Could not locate heading '{prefix}' inside Pass 1 PDF pages")
        page_map[b["title"]] = f"{found_pg:02d}"

    html2 = _build_html_doc(title, blocks, page_map=page_map)
    pages2 = _render_pdf(html2, OUT_PDF)
    full_txt = "\n".join(pages2)

    # Strict regression sanity assertions on extracted PDF text
    for bad in ("9990", "B1,1", "Spread ost", "*ZeroFit", "**"):
        if bad in full_txt:
            raise AssertionError(f"Forbidden PDF regression artifact found: {bad}")
    if re.search(r"\b(?:ut|ost)\b", full_txt):
        raise AssertionError("Truncated word token ('ut'/'ost') detected in PDF text")
    for key in ("15.61%", "28.78%", "12.95%", "Sec 50AA", "Parameter Provenance"):
        if key not in full_txt:
            raise AssertionError(f"Expected synchronized spec token missing from PDF: {key}")

    print(f"Built clean PDF ({len(pages2)} pages, TOC synced) -> {OUT_PDF}")
    return OUT_PDF


if __name__ == "__main__":
    build_pdf()
