#!/usr/bin/env python
"""
Minimal Markdown -> PDF converter for the GOAI Track-3 submission docs.

Handles (enough for our docs):
  #/##/###/#### headings, pipe tables (with **bold** cells), > blockquotes,
  ``` fenced code (ASCII diagrams), - / * bullets (nested via indent),
  1. ordered lists, --- horizontal rules, and **bold** / *italic* / `code`
  inline spans in paragraphs.

CJK-aware: uses wrapmode="CHAR" + markdown=True so Chinese wraps correctly and
inline emphasis renders. Requires a CJK TTF (Noto Sans SC) + a heavier face
(simhei) for bold/headings.
"""
import sys, re, os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

REGULAR = r"C:/Windows/Fonts/Noto Sans SC.ttf"
BOLD    = r"C:/Windows/Fonts/simhei.ttf"

CJK_RE = re.compile(r"[\u2E80-\u9FFF\u3000-\u303F\uFF00-\uFFEF\u3400-\u4DBF"
                    r"\uF900-\uFAFF\u2460-\u24FF\u25A0-\u25FF\u2713\u2717]")

# Glyphs absent from Noto Sans SC (regular) / SimHei (bold) -> safe substitutes.
GLYPH_FIX = str.maketrans({"►": "→", "◄": "←", "✗": "×"})

def _fix(s):
    return s.translate(GLYPH_FIX)

def is_cjk(ch):
    return bool(CJK_RE.match(ch))

def strip_md(s):
    return s.replace("**", "").replace("`", "")

class CJKDoc(FPDF):
    def __init__(self, regular, bold):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.add_font("CJK", "", regular)
        self.add_font("CJK", "B", bold)
        self.set_margins(18, 18, 18)
        self.set_title("GOAI Track3 submission")

    # ---- low-level helpers -------------------------------------------------
    def _measure(self, txt, size, style=""):
        self.set_font("CJK", style, size)
        return self.get_string_width(strip_md(txt))

    def h_rule(self):
        self.ln(1.5)
        y = self.get_y()
        self.set_draw_color(190, 190, 190)
        self.set_line_width(0.3)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

    # ---- block renderers ---------------------------------------------------
    def heading(self, level, text):
        text = _fix(text)
        sizes = {1: 16, 2: 13, 3: 11, 4: 10}
        sz = sizes.get(level, 10)
        self.ln(2 if level <= 2 else 1.5)
        self.set_font("CJK", "B", sz)
        self.set_text_color(20, 30, 60)
        self.multi_cell(0, sz * 0.55 + 2, text, new_x=XPos.LMARGIN,
                        new_y=YPos.NEXT, wrapmode="CHAR")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def paragraph(self, text, size=9.5, indent=0):
        text = _fix(text)
        self.set_font("CJK", "", size)
        self.set_x(self.l_margin + indent)
        w = self.w - self.r_margin - (self.l_margin + indent)
        self.multi_cell(w, 5.0, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                        wrapmode="CHAR", markdown=True, align="L")
        self.ln(0.8)

    def blockquote(self, text):
        text = _fix(text)
        self.ln(1)
        self.set_fill_color(244, 246, 250)
        self.set_draw_color(90, 130, 200)
        self.set_line_width(0.6)
        x = self.l_margin + 2
        self.set_font("CJK", "", 9.3)
        # measure height
        self.set_xy(x + 3, self.get_y())
        # render text first to know height
        y0 = self.get_y()
        self.multi_cell(self.w - self.r_margin - x - 3, 5.0, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT, wrapmode="CHAR",
                        markdown=True)
        y1 = self.get_y()
        h = y1 - y0
        self.set_fill_color(244, 246, 250)
        self.rect(x, y0 - 1, self.w - self.r_margin - x, h + 2, style="F")
        self.set_xy(x + 1, y0 - 1)
        self.set_draw_color(90, 130, 200)
        self.line(x, y0 - 1, x, y0 + h + 1)
        # re-render text on top
        self.set_xy(x + 3, y0)
        self.multi_cell(self.w - self.r_margin - x - 4, 5.0, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT, wrapmode="CHAR",
                        markdown=True)
        self.ln(1.5)

    def code_block(self, text):
        text = _fix(text)
        self.ln(1)
        self.set_font("CJK", "", 7.2)
        self.set_fill_color(238, 238, 238)
        for line in text.split("\n"):
            disp = line.replace(" ", "\u00a0")  # preserve indentation
            self.set_x(self.l_margin)
            self.multi_cell(0, 3.6, disp, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                            wrapmode="CHAR", fill=True)
        self.set_fill_color(255, 255, 255)
        self.ln(1.5)

    def bullet(self, text, level=0, ordered=None):
        text = _fix(text)
        indent = 4 + level * 5
        size = 9.3
        self.set_font("CJK", "", size)
        marker = ("•" if ordered is None else f"{ordered}.")
        self.set_x(self.l_margin + indent)
        # marker column (render with regular face so the bullet glyph exists)
        mw = 5
        self.set_font("CJK", "", size)
        self.cell(mw, 4.8, marker)
        self.set_font("CJK", "", size)
        w = self.w - self.r_margin - (self.l_margin + indent + mw)
        self.multi_cell(w, 4.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                        wrapmode="CHAR", markdown=True)
        self.ln(0.4)

    def table(self, rows, size=8.6, line_h=4.3):
        # strip & parse
        data = [[_fix(c.strip()) for c in r.split("|")] for r in rows]
        data = [[c for c in r if c != ""] for r in data]  # drop empty edge cells
        ncol = max(len(r) for r in data)
        data = [r + [""] * (ncol - len(r)) for r in data]
        usable = self.w - self.l_margin - self.r_margin
        # raw widths
        raw = [0.0] * ncol
        for r in data:
            for c in range(ncol):
                raw[c] = max(raw[c], self._measure(r[c], size))
        total = sum(raw) or 1.0
        colw = [max(raw[c], usable * 0.10) for c in range(ncol)]
        s = sum(colw)
        if s > usable:
            colw = [w * usable / s for w in colw]
        # render
        x0 = self.l_margin
        for ri, row in enumerate(data):
            is_head = (ri == 0)
            cell_lines = []
            for c in range(ncol):
                self.set_font("CJK", "B" if is_head else "", size)
                lines = self._wrap_cell(row[c], colw[c], size,
                                        "B" if is_head else "")
                cell_lines.append(lines)
            nlines = max(len(l) for l in cell_lines)
            row_h = nlines * line_h
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
            y = self.get_y()
            self.set_font("CJK", "B" if is_head else "", size)
            if is_head:
                self.set_fill_color(223, 231, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for c in range(ncol):
                cx = x0 + sum(colw[:c])
                padded = cell_lines[c] + [""] * (nlines - len(cell_lines[c]))
                self.set_xy(cx, y)
                self.multi_cell(colw[c], line_h, "\n".join(padded), border=1,
                                new_x=XPos.RIGHT, new_y=YPos.TOP,
                                wrapmode="CHAR", fill=True, align="L",
                                markdown=True)
            self.set_xy(x0, y + row_h)
        self.ln(2)

    def _wrap_cell(self, txt, width, size, style=""):
        self.set_font("CJK", style, size)
        words = list(txt)
        lines, cur = [], ""
        for ch in words:
            trial = cur + ch
            if self.get_string_width(strip_md(trial)) > width - 1.2 and cur:
                lines.append(cur)
                cur = ch
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines or [""]


def is_table_row(line):
    s = line.strip()
    return "|" in s and not s.startswith("```")

def is_table_sep(line):
    s = line.strip().strip("|").replace(" ", "")
    return bool(s) and set(s) <= set("-:") and "-" in s

def parse_and_render(pdf, md_text):
    lines = md_text.split("\n")
    i = 0
    n = len(lines)
    list_stack = []  # (level, kind) for ordered numbering
    while i < n:
        line = lines[i]
        raw = line
        stripped = line.strip()
        # code fence
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1  # skip closing fence
            pdf.code_block("\n".join(buf))
            list_stack.clear()
            continue
        # blank
        if stripped == "":
            pdf.ln(1.5)
            list_stack.clear()
            i += 1
            continue
        # horizontal rule
        if set(stripped) <= set("-") and len(stripped) >= 3:
            pdf.h_rule(); list_stack.clear(); i += 1; continue
        # table
        if is_table_row(line) and i + 1 < n and is_table_sep(lines[i + 1]):
            tbl = [line]
            i += 1
            while i < n and is_table_row(lines[i]):
                tbl.append(lines[i]); i += 1
            pdf.table(tbl)
            list_stack.clear()
            continue
        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            pdf.heading(len(m.group(1)), m.group(2).strip())
            list_stack.clear(); i += 1; continue
        # blockquote
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            pdf.blockquote("\n".join(buf))
            list_stack.clear(); continue
        # ordered list
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            pdf.bullet(m.group(3), level=indent, ordered=int(m.group(2)))
            i += 1; continue
        # bullet list
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            pdf.bullet(m.group(2), level=indent)
            i += 1; continue
        # paragraph
        pdf.paragraph(stripped)
        list_stack.clear()
        i += 1


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"
    with open(src, encoding="utf-8") as f:
        md = f.read()
    pdf = CJKDoc(REGULAR, BOLD)
    pdf.add_page()
    parse_and_render(pdf, md)
    pdf.output(out)
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
