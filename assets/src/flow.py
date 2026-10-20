"""A small design system for the course diagrams.

Every figure is drawn on a 900 px wide canvas. That width matches a GitHub / Colab content column,
so the type sizes below are (roughly) the sizes a reader actually sees. Text is measured with the
real font metrics: boxes grow to fit their words, and text is never shrunk to fit a box.

Each figure is written as an SVG with the Inter font (SIL Open Font License) embedded as a small
subset, plus a 2x PNG for the notebooks. Rendering the PNG needs Google Chrome.
"""
import base64
import html
import io
import os
import subprocess
import urllib.request

# --- design tokens ------------------------------------------------------------------------------
WIDTH = 900
MARGIN = 48
PAD = 22                 # padding inside a node
RADIUS = 14              # node corner radius
ARROW_W = 2.25

SIZE = {"title": 30, "subtitle": 19, "node": 23, "sub": 19, "label": 18, "small": 17}
WEIGHT = {"title": 700, "subtitle": 500, "node": 600, "sub": 500, "label": 600, "small": 500}

INK = {
    "bg": "#FFFFFF", "text": "#17202A", "text2": "#56616D", "border": "#DDE2E7", "surface": "#F5F7F9",
    "accent": "#2563EB", "accent_bg": "#EFF6FF", "accent_ink": "#1D4ED8",
    "success": "#16803C", "success_bg": "#EEF7F1", "failure": "#C2413A", "failure_bg": "#FCEFEE",
    "arrow": "#7B8794", "muted_mark": "#B4BCC4",
}
FAMILY = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# node style -> (fill, stroke, stroke width, dashed)
STYLE = {
    "plain":   ("#FFFFFF", INK["border"], 1.6, False),
    "surface": (INK["surface"], INK["border"], 1.6, False),
    "accent":  (INK["accent_bg"], INK["accent"], 2.2, False),
    "success": (INK["success_bg"], INK["success"], 2.0, False),
    "failure": (INK["failure_bg"], INK["failure"], 2.0, False),
    "ghost":   ("#FFFFFF", INK["arrow"], 1.6, True),
}

# --- fonts: download once, measure text, embed subsets ------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
FONT_CSS = "https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700"
_fonts = {}


def _font_path(weight):
    path = os.path.join(FONT_DIR, f"Inter-{weight}.ttf")
    if not os.path.exists(path):
        os.makedirs(FONT_DIR, exist_ok=True)
        css = urllib.request.urlopen(FONT_CSS).read().decode()
        urls = [part.split(")")[0] for part in css.split("url(")[1:]]
        for w, url in zip((500, 600, 700), urls):
            urllib.request.urlretrieve(url, os.path.join(FONT_DIR, f"Inter-{w}.ttf"))
    return path


def _font(weight):
    if weight not in _fonts:
        from fontTools.ttLib import TTFont
        f = TTFont(_font_path(weight))
        _fonts[weight] = (f, f.getBestCmap(), f["hmtx"], f["head"].unitsPerEm)
    return _fonts[weight]


def measure(s, size, weight=600):
    """Width of a string in px, from the font's real advance widths."""
    _, cmap, hmtx, upm = _font(weight)
    return sum(hmtx[cmap.get(ord(ch), cmap[ord("?")])][0] for ch in s) * size / upm


def wrap(s, size, weight, max_w):
    """Greedy word wrap that never shrinks the font."""
    out, line = [], ""
    for word in s.split(" "):
        trial = f"{line} {word}".strip()
        if measure(trial, size, weight) <= max_w or not line:
            line = trial
        else:
            out.append(line)
            line = word
    return out + [line]


def _font_face_css(chars_by_weight):
    from fontTools import subset
    css = ""
    for weight, chars in sorted(chars_by_weight.items()):
        opts = subset.Options()
        opts.flavor = "woff2"
        opts.layout_features = ["kern", "liga", "calt", "tnum"]
        sub = subset.Subsetter(opts)
        sub.populate(text="".join(sorted(chars | {" "})))
        from fontTools.ttLib import TTFont
        font = TTFont(_font_path(weight))
        sub.subset(font)
        buf = io.BytesIO()
        font.flavor = "woff2"
        font.save(buf)
        data = base64.b64encode(buf.getvalue()).decode()
        css += (f'@font-face{{font-family:"Inter";font-weight:{weight};'
                f'src:url(data:font/woff2;base64,{data}) format("woff2");}}')
    return css


# --- the canvas ---------------------------------------------------------------------------------
class Canvas:
    def __init__(self, height, width=WIDTH):
        self.w, self.h = width, height
        self.parts = []
        self.chars = {}
        self._markers = {}
        self.boxes = []          # text extents, checked against the canvas edges on save

    # text --------------------------------------------------------------------------------------
    def text(self, x, y, s, role="sub", color=None, anchor="start", weight=None, size=None):
        size = size or SIZE[role]
        weight = weight or WEIGHT[role]
        color = color or (INK["text"] if role in ("title", "node", "label") else INK["text2"])
        self._extent(x, s, size, weight, anchor)
        self.chars.setdefault(weight, set()).update(s)
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{html.escape(s)}</text>')

    def halo_text(self, x, y, s, role="label", color=None, anchor="middle"):
        """Text with a white halo, so it can sit on top of a line and stay readable."""
        size, weight = SIZE[role], WEIGHT[role]
        color = color or INK["text"]
        self._extent(x, s, size, weight, anchor)
        self.chars.setdefault(weight, set()).update(s)
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central" stroke="#FFFFFF" stroke-width="7" '
            f'stroke-linejoin="round" paint-order="stroke">{html.escape(s)}</text>')

    def _extent(self, x, s, size, weight, anchor):
        w = measure(s, size, weight)
        left = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        self.boxes.append((left, left + w, s))

    def check(self, margin=16):
        bad = [s for l, r, s in self.boxes if l < margin or r > self.w - margin]
        assert not bad, f"text outside the canvas: {bad}"

    def step(self, x, y, n, s, anchor="middle", halo=False):
        """A numbered step label: a small dark badge followed by bold text."""
        tw = measure(s, SIZE["label"], WEIGHT["label"])
        total = 26 + 10 + tw
        left = x - total / 2 if anchor == "middle" else (x if anchor == "start" else x - total)
        if halo:
            self.parts.append(f'<rect x="{left - 6:.1f}" y="{y - 17:.1f}" width="{total + 12:.1f}" height="34" rx="8" fill="#FFFFFF"/>')
        self.parts.append(f'<circle cx="{left + 13:.1f}" cy="{y:.1f}" r="13" fill="{INK["text2"]}"/>')
        self.text(left + 13, y + 0.5, str(n), "small", color="#FFFFFF", anchor="middle", weight=700, size=15)
        self.text(left + 36, y, s, "label", anchor="start")

    def header(self, title, subtitle=None):
        self.text(MARGIN, 58, title, "title")
        if subtitle:
            self.text(MARGIN, 96, subtitle, "subtitle")
        return 140 if subtitle else 110            # y where content may start

    # shapes ------------------------------------------------------------------------------------
    def rect(self, x, y, w, h, style="plain", rx=RADIUS):
        fill, stroke, sw, dashed = STYLE[style]
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')

    def node(self, x, y, title, sub=None, style="plain", w=None, align="center", min_h=None):
        """A box sized to its text. Returns (x, y, w, h). Long text wraps; it never shrinks."""
        max_inner = (w - 2 * PAD) if w else 10_000
        t_lines = wrap(title, SIZE["node"], WEIGHT["node"], max_inner)
        s_lines = wrap(sub, SIZE["sub"], WEIGHT["sub"], max_inner) if sub else []
        if not w:
            w = 2 * PAD + max([measure(l, SIZE["node"], WEIGHT["node"]) for l in t_lines] +
                              [measure(l, SIZE["sub"], WEIGHT["sub"]) for l in s_lines])
        lh_t, lh_s = SIZE["node"] * 1.3, SIZE["sub"] * 1.35
        content = len(t_lines) * lh_t + (8 + len(s_lines) * lh_s if s_lines else 0)
        h = max(min_h or 0, content + 2 * 16)
        self.rect(x, y, w, h, style)
        ty = y + (h - content) / 2 + lh_t / 2
        tx, anchor = (x + w / 2, "middle") if align == "center" else (x + PAD, "start")
        for line in t_lines:
            self.text(tx, ty, line, "node", anchor=anchor)
            ty += lh_t
        ty += 8 - lh_t / 2 + lh_s / 2
        for line in s_lines:
            self.text(tx, ty, line, "sub", anchor=anchor)
            ty += lh_s
        return x, y, w, h

    def diamond(self, cx, cy, w, h, label, style="plain"):
        fill, stroke, sw, _ = STYLE[style]
        pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        self.parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                          f'stroke-linejoin="round"/>')
        lines = wrap(label, SIZE["node"] - 2, WEIGHT["node"], w * 0.62)
        lh = (SIZE["node"] - 2) * 1.25
        for i, line in enumerate(lines):
            self.text(cx, cy + (i - (len(lines) - 1) / 2) * lh, line, "node", anchor="middle", size=SIZE["node"] - 2)

    def circle(self, cx, cy, r, label, filled=False):
        fill, stroke, color = (INK["accent"], INK["accent"], "#FFFFFF") if filled else ("#FFFFFF", INK["border"], INK["text"])
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.text(cx, cy + 1, label, "node", color=color, anchor="middle", weight=700, size=21)

    def line(self, pts, color=None, width=ARROW_W, dashed=False, arrow=True):
        color = color or INK["arrow"]
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        marker = f' marker-end="url(#{self._marker(color)})"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                          f'stroke-linejoin="round" stroke-linecap="round"{dash}{marker}/>')

    def raw(self, svg):
        self.parts.append(svg)

    def _marker(self, color):
        mid = "m" + color.strip("#")
        self._markers[mid] = color
        return mid

    # output ------------------------------------------------------------------------------------
    def svg(self, alt):
        markers = "".join(
            f'<marker id="{mid}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="11" markerHeight="11" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M1 1.5 L11 6 L1 10.5 Z" fill="{c}"/></marker>'
            for mid, c in self._markers.items())
        css = _font_face_css(self.chars) + f"text{{font-family:{FAMILY};font-kerning:normal}}"
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{html.escape(alt)}">'
                f'<title>{html.escape(alt)}</title><style>{css}</style><defs>{markers}</defs>'
                f'<rect x="0.75" y="0.75" width="{self.w - 1.5}" height="{self.h - 1.5}" rx="18" '
                f'fill="#FFFFFF" stroke="{INK["border"]}" stroke-width="1.5"/>'
                + "".join(self.parts) + "</svg>")


CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def save(canvas, name, alt, outdir):
    """Write <name>.svg (fonts embedded) and a 2x <name>.png rendered from it."""
    canvas.check()
    os.makedirs(outdir, exist_ok=True)
    svg_path = os.path.abspath(os.path.join(outdir, f"{name}.svg"))
    png_path = os.path.abspath(os.path.join(outdir, f"{name}.png"))
    with open(svg_path, "w") as f:
        f.write(canvas.svg(alt))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--default-background-color=00000000", f"--window-size={canvas.w},{canvas.h}",
                    f"--screenshot={png_path}", f"file://{svg_path}"], check=True, capture_output=True)
    return svg_path, png_path
