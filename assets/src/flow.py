"""Standard flowchart toolkit: white background, clean boxes, labeled arrows, a validated palette."""
import html
import os
import subprocess
import tempfile

# Validated categorical palette (light mode, white surface) + darker shades for text on tints.
COL = {
    "blue":   {"c": "#2a78d6", "tint": "#eaf2fc", "ink": "#1d5aa6"},
    "orange": {"c": "#eb6834", "tint": "#fdeee7", "ink": "#b5461b"},
    "green":  {"c": "#1baf7a", "tint": "#e5f6ef", "ink": "#11714f"},
    "violet": {"c": "#4a3aa7", "tint": "#edebf8", "ink": "#3a2d8a"},
    "red":    {"c": "#e34948", "tint": "#fcebeb", "ink": "#b02f2e"},
    "yellow": {"c": "#eda100", "tint": "#fdf4dd", "ink": "#8a5d00"},
    "grey":   {"c": "#8c959f", "tint": "#f6f8fa", "ink": "#424a53"},
}
TEXT, TEXT2, BORDER, ARROW = "#1f2328", "#57606a", "#d0d7de", "#57606a"
FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=block"
SANS, MONO = "Inter", "JetBrains Mono"


def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=15, color=TEXT, weight=600, anchor="middle", font=SANS, extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" dominant-baseline="central" {extra}>{esc(s)}</text>')


def multiline(x, y, rows, size=15, color=TEXT, weight=600, anchor="middle", font=SANS, gap=1.35):
    """Rows centered vertically on y."""
    top = y - (len(rows) - 1) * size * gap / 2
    return "".join(text(x, top + i * size * gap, r, size, color, weight, anchor, font) for i, r in enumerate(rows))


def box(x, y, w, h, color="blue", rows=(), sub=(), style="tint", rx=10, size=15, sub_font=SANS, sub_size=12.5):
    """style: 'tint' (light fill, colored border), 'solid' (filled, white text), 'plain' (white, grey border)."""
    c = COL[color]
    if style == "solid":
        fill, stroke, fg, fg2 = c["c"], c["c"], "#ffffff", "#ffffff"
    elif style == "plain":
        fill, stroke, fg, fg2 = "#ffffff", BORDER, TEXT, TEXT2
    else:
        fill, stroke, fg, fg2 = c["tint"], c["c"], c["ink"], TEXT2
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    rows, sub = list(rows), list(sub)
    if sub:
        n_main, n_sub = len(rows), len(sub)
        total = n_main * size * 1.3 + n_sub * sub_size * 1.4 + 6
        top = y + h / 2 - total / 2
        for i, r in enumerate(rows):
            out += text(x + w / 2, top + size * 0.65 + i * size * 1.3, r, size, fg, 700)
        base = top + n_main * size * 1.3 + 6
        for i, r in enumerate(sub):
            out += text(x + w / 2, base + sub_size * 0.7 + i * sub_size * 1.4, r, sub_size, fg2, 500, font=sub_font)
    else:
        out += multiline(x + w / 2, y + h / 2, rows, size, fg, 700)
    return out


def pill(x, y, w, h, color, label, style="solid", size=15):
    return box(x, y, w, h, color, [label], style=style, rx=h / 2, size=size)


def diamond(cx, cy, w, h, label_rows, color="yellow", size=14):
    c = COL[color]
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return (f'<polygon points="{pts}" fill="{c["tint"]}" stroke="{c["c"]}" stroke-width="2" stroke-linejoin="round"/>'
            + multiline(cx, cy, label_rows, size, c["ink"], 700))


def markers():
    out = ""
    for name, c in list(COL.items()) + [("arrow", {"c": ARROW})]:
        out += (f'<marker id="m-{name}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="8" markerHeight="8" '
                f'orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0.8 L10 5 L0 9.2 Z" fill="{c["c"]}"/></marker>')
    return out


def arrow(points, color="arrow", width=2, dashed=False, curve=None):
    """Orthogonal polyline (list of points) or a quadratic curve via `curve` control point."""
    stroke = ARROW if color == "arrow" else COL[color]["c"]
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    if curve:
        (x1, y1), (x2, y2) = points
        d = f"M{x1} {y1} Q{curve[0]} {curve[1]} {x2} {y2}"
    else:
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
    return (f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"'
            f'{dash} marker-end="url(#m-{color})"/>')


def label(x, y, s, color=TEXT2, size=13, weight=600, font=SANS, bg="#ffffff", pad=6):
    """Arrow label on a white background so it sits cleanly over lines."""
    w = len(s) * size * (0.6 if font == MONO else 0.5) + 2 * pad
    return (f'<rect x="{x - w / 2}" y="{y - size * 0.85}" width="{w}" height="{size * 1.7}" rx="4" fill="{bg}"/>'
            + text(x, y, s, size, color, weight, font=font))


def note(x, y, s, color=TEXT2, size=13, weight=600, anchor="middle", font=SANS):
    """A label that sits just above a line (no background, the line stays continuous)."""
    return text(x, y - size * 0.95, s, size, color, weight, anchor, font)


def badge(x, y, n, color="violet", r=12):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL[color]["c"]}"/>'
            + text(x, y + 0.5, n, r * 1.05, "#ffffff", 700))


def title(s, sub=None):
    out = text(36, 40, s, 22, TEXT, 700, "start")
    if sub:
        out += text(36, 68, sub, 14.5, TEXT2, 500, "start")
    return out


def page(w, h, body):
    return f"""<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{FONTS}">
<style>html,body{{margin:0;background:transparent}} svg{{display:block}}</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>{markers()}
<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="7" height="7" fill="{COL['orange']['tint']}"/><rect width="2" height="7" fill="{COL['orange']['c']}" opacity=".22"/></pattern>
</defs>
<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="14" fill="#ffffff" stroke="{BORDER}" stroke-width="1.5"/>
{body}</svg></body></html>"""


CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def render(name, w, h, body, outdir):
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(tempfile.mkdtemp(), f"{name}.html")   # intermediate page, kept out of assets/
    open(path, "w").write(page(w, h, body))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--default-background-color=00000000", f"--window-size={w},{h}", "--virtual-time-budget=8000",
                    f"--screenshot={outdir}/{name}.png", f"file://{path}"], check=True, capture_output=True)
    return f"{outdir}/{name}.png"
