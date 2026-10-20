"""Regenerate every course diagram:   python assets/src/diagrams.py [out_dir] [names...]

Writes <name>.svg (README, crisp at any size) and <name>.png (2x, embedded in the notebooks).
Needs fontTools + brotli (`pip install fonttools brotli`) and Google Chrome (set CHROME=... if it is
not in the default macOS location). The Inter font is downloaded once into assets/src/fonts/.

One figure = one idea. If a figure gets crowded, split it rather than shrinking the text.
"""
import json
import os
import sys

from flow import INK, MARGIN, SIZE, WEIGHT, WIDTH, Canvas, measure, save

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = {}


def figure(name, alt):
    def register(fn):
        FIGURES[name] = (fn, alt)
        return fn
    return register


def below(box, gap=0):
    x, y, w, h = box
    return x + w / 2, y + h + gap


def above(box, gap=0):
    x, y, w, h = box
    return x + w / 2, y - gap


def right(box, gap=0):
    x, y, w, h = box
    return x + w + gap, y + h / 2


def left(box, gap=0):
    x, y, w, h = box
    return x - gap, y + h / 2


# ------------------------------------------------------------------------------------------------
@figure("ladder", "Learning path with four stages: 1 Q-learning, learn from rewards. 2 REINFORCE, learn which "
                  "actions are more likely to work. 3 GRPO, compare several attempts. 4 Agentic RL, train an LLM to use tools.")
def ladder():
    stages = [("Q-learning", "Learn from rewards."),
              ("REINFORCE", "Learn which actions are more likely to work."),
              ("GRPO", "Compare several attempts at the same task."),
              ("Agentic RL", "Train a real LLM to use tools.")]
    card_h, gap = 86, 20
    c = Canvas(140 + 4 * card_h + 3 * gap + 44)
    y0 = c.header("Your learning path", "Four chapters. Each one adds a single new idea.")
    cx = MARGIN + 24
    c.line([(cx, y0 + card_h / 2), (cx, y0 + 3 * (card_h + gap) + card_h / 2)], INK["border"], 3, arrow=False)
    for i, (name, desc) in enumerate(stages):
        y = y0 + i * (card_h + gap)
        last = i == len(stages) - 1
        c.rect(cx + 44, y, WIDTH - MARGIN - (cx + 44), card_h, "accent" if last else "plain")
        c.circle(cx, y + card_h / 2, 24, str(i + 1), filled=last)
        c.text(cx + 70, y + card_h / 2 - 15, name, "node")
        c.text(cx + 70, y + card_h / 2 + 17, desc, "sub")
        c.text(WIDTH - MARGIN - 24, y + card_h / 2 - 15, f"Chapter {i + 1}", "small", anchor="end")
    return c


# ------------------------------------------------------------------------------------------------
@figure("rl_loop", "Reinforcement learning loop: the agent takes an action (use the calculator), the environment "
                   "sends feedback (56 is correct, reward +1), the agent learns what worked and tries again.")
def rl_loop():
    c = Canvas(560)
    c.header("How reinforcement learning works", "The agent tries something, the world reacts, and the agent learns.")
    agent = c.node(60, 176, "Agent", "decides what to do", "accent", w=260, min_h=104)
    env = c.node(580, 176, "Environment", "reacts to the action", "plain", w=260, min_h=104)
    learn = c.node(580, 392, "Agent learns", "keeps what worked", "surface", w=260, min_h=104)
    # 1. action
    (x1, y1), (x2, _) = right(agent, 4), left(env, 6)
    c.line([(x1, y1), (x2, y1)])
    c.step((x1 + x2) / 2, y1 - 26, 1, "Action")
    c.text((x1 + x2) / 2, y1 + 25, "use the calculator", "sub", anchor="middle")
    # 2. feedback
    (fx, fy), (_, ly) = below(env, 4), above(learn, 6)
    c.line([(fx, fy), (fx, ly)])
    c.step(fx - 18, (fy + ly) / 2 - 15, 2, "Feedback", anchor="end")
    c.text(fx - 18, (fy + ly) / 2 + 16, "“56 is correct” · reward +1", "sub", anchor="end")
    # 3. try again
    (bx, by), (ax, ay) = left(learn, 4), below(agent, 6)
    c.line([(bx, by), (ax, by), (ax, ay)])
    c.step((bx + ax) / 2 + 30, by - 26, 3, "Try again")
    c.text((bx + ax) / 2 + 30, by + 26, "a little smarter each time", "sub", anchor="middle")
    return c


# ------------------------------------------------------------------------------------------------
@figure("agentic_task", "Flowchart of an agent solving 'What is 37 times 48?': read the question, choose a tool "
                        "(calculator), use the tool, read the result (1,776), decide whether it has enough to answer; "
                        "if not, choose a tool again; if yes, finish with the answer.")
def agentic_task():
    c = Canvas(930)
    c.header("An agent solving a task with a tool", "Example: “What is 37 × 48?”")
    x, w, gap = 140, 400, 34
    cx = x + w / 2
    steps = [("Read the question", "What is 37 × 48?", "plain"),
             ("Choose a tool", "the calculator", "accent"),
             ("Use the tool", "calculator(37 × 48)", "plain"),
             ("Read the result", "1,776", "plain")]
    boxes, y = [], 140
    for title, sub, style in steps:
        b = c.node(x, y, title, sub, style, w=w)
        if boxes:
            c.line([below(boxes[-1], 4), (cx, y - 6)])
        boxes.append(b)
        y += b[3] + gap
    dy = y + 60
    c.line([below(boxes[-1], 4), (cx, dy - 60 - 6)])
    c.diamond(cx, dy, 300, 120, "Enough to answer?")
    fin = c.node(x, dy + 60 + gap + 10, "Finish", "“The answer is 1,776.”", "success", w=w)
    c.line([(cx, dy + 60 + 4), (cx, fin[1] - 6)])
    c.halo_text(cx + 26, dy + 60 + (gap + 10) / 2, "yes", "label", anchor="start")
    # retry branch on the right
    rx = x + w + 90
    ty = boxes[1][1] + boxes[1][3] / 2
    c.line([(cx + 150 + 4, dy), (rx, dy), (rx, ty), (x + w + 6, ty)])
    c.text(rx + 16, (dy + ty) / 2 - 14, "no:", "label")
    c.text(rx + 16, (dy + ty) / 2 + 14, "get more", "sub")
    c.text(rx + 16, (dy + ty) / 2 + 40, "information", "sub")
    return c


@figure("agentic_learning", "Learning cycle: the completed task is checked, the agent receives a reward, and its "
                            "future decisions improve before the next task.")
def agentic_learning():
    c = Canvas(500)
    c.header("How the agent learns from the whole task", "One reward at the end shapes every decision it made.")
    w, x0, x1, r0, r1 = 330, 60, 510, 150, 330
    tl = c.node(x0, r0, "Completed task", "all the steps it took", "plain", w=w, min_h=100)
    tr = c.node(x1, r0, "Check the outcome", "Is 1,776 correct?", "plain", w=w, min_h=100)
    br = c.node(x1, r1, "Reward", "+1 if correct, −1 if not", "surface", w=w, min_h=100)
    bl = c.node(x0, r1, "Better decisions", "repeat what worked", "accent", w=w, min_h=100)
    c.line([right(tl, 4), left(tr, 6)])
    c.line([below(tr, 4), above(br, 6)])
    c.line([left(br, 4), right(bl, 6)])
    c.line([above(bl, 4), below(tl, 6)])
    c.text(x0 + w / 2 + 18, (r0 + 100 + r1) / 2, "next task", "label")
    return c


# ------------------------------------------------------------------------------------------------
@figure("two_states", "Q-learning success path: needs information, uses the right tool (cost 0.02), has the "
                      "information, finishes (+1), success. Mistakes listed separately: wrong tool -0.15, finishing "
                      "too early -1, more than 4 moves -0.5.")
def two_states():
    c = Canvas(650)
    c.header("Q-learning: the agent's two situations", "It needs information from the right tool before it can finish.")
    x, w = 60, 290
    n1 = c.node(x, 150, "Needs information", "no answer yet", "plain", w=w, min_h=96)
    n2 = c.node(x, 326, "Has the information", "a tool returned it", "plain", w=w, min_h=96)
    n3 = c.node(x, 502, "Success", "reward +1", "success", w=w, min_h=96)
    for a, b, act, pts in [(n1, n2, "uses the right tool", "−0.02 points"), (n2, n3, "presses Finish", "+1 point")]:
        (ax, ay), (_, by) = below(a, 4), above(b, 6)
        c.line([(ax, ay), (ax, by)], INK["success"], 2.6)
        c.text(ax + 20, (ay + by) / 2 - 13, act, "label")
        c.text(ax + 20, (ay + by) / 2 + 14, pts, "sub")
    # mistakes, kept away from the main path
    px = 520
    c.raw(f'<line x1="{px - 40}" y1="150" x2="{px - 40}" y2="598" stroke="{INK["border"]}" stroke-width="1.5"/>')
    c.text(px, 166, "Mistakes cost points", "node")
    y = 204
    for title, sub, style in [("Uses the wrong tool", "still no answer · −0.15", "plain"),
                              ("Finishes too early", "the task fails · −1", "failure"),
                              ("Takes more than 4 moves", "time is up · −0.5", "plain")]:
        b = c.node(px, y, title, sub, style, w=WIDTH - MARGIN - px, align="left")
        y += b[3] + 16
    c.text(MARGIN, 628, "In RL words: a situation is a state, and pressing a button is an action.", "small")
    return c


# ------------------------------------------------------------------------------------------------
@figure("reinforce", "REINFORCE: try an episode, get a reward, compare it with the usual reward. Better than usual: "
                     "make those choices more likely. Worse than usual: make those choices less likely.")
def reinforce():
    c = Canvas(730)
    c.header("REINFORCE: learn from better-than-usual attempts", "Was this attempt better or worse than usual?")
    w = 440
    x = (WIDTH - w) / 2
    n1 = c.node(x, 140, "Try an episode", "press buttons until Finish", "plain", w=w)
    n2 = c.node(x, n1[1] + n1[3] + 40, "Get a reward", "for example +0.98", "plain", w=w)
    n3 = c.node(x, n2[1] + n2[3] + 40, "Compare with usual", "usual = the average of recent rewards", "accent", w=w)
    c.line([below(n1, 4), above(n2, 6)])
    c.line([below(n2, 4), above(n3, 6)])
    fy = n3[1] + n3[3] + 34
    cw = 360
    lx, rx = 60, WIDTH - 60 - cw
    top = fy + 40
    c.line([below(n3, 4), (WIDTH / 2, fy)], arrow=False)
    c.line([(WIDTH / 2, fy), (lx + cw / 2, fy), (lx + cw / 2, top - 6)], INK["success"], 2.6)
    c.line([(WIDTH / 2, fy), (rx + cw / 2, fy), (rx + cw / 2, top - 6)], INK["failure"], 2.6)
    c.node(lx, top, "Better than usual", "↑ make those choices more likely", "success", w=cw)
    c.node(rx, top, "Worse than usual", "↓ make those choices less likely", "failure", w=cw)
    return c


# ------------------------------------------------------------------------------------------------
@figure("grpo_group", "GRPO with four attempts at 'What is 37 times 48?': attempts A and C are wrong (reward 0), "
                      "B and D are correct (reward 1). Compare the attempts, then update the policy so B and D "
                      "become more likely and A and C less likely.")
def grpo_group():
    c = Canvas(790)
    c.header("GRPO: compare several attempts", "A simplified picture with 4 attempts. In the notebooks each question gets 8.")
    qw = 400
    q = c.node((WIDTH - qw) / 2, 140, "What is 37 × 48?", None, "surface", w=qw, min_h=64)
    attempts = [("A", "1,676", False), ("B", "1,776", True), ("C", "1,786", False), ("D", "1,776", True)]
    cw, gap = 186, 20
    x0 = (WIDTH - (4 * cw + 3 * gap)) / 2
    top, ch = 268, 160
    bus_y = top - 34
    qx, qy = below(q, 4)
    c.line([(qx, qy), (qx, bus_y)], arrow=False)
    c.raw(f'<line x1="{x0 + cw / 2}" y1="{bus_y}" x2="{x0 + 3 * (cw + gap) + cw / 2}" y2="{bus_y}" '
          f'stroke="{INK["arrow"]}" stroke-width="2.25"/>')
    bottom_bus = top + ch + 34
    for i, (name, answer, ok) in enumerate(attempts):
        x = x0 + i * (cw + gap)
        mx = x + cw / 2
        c.line([(mx, bus_y), (mx, top - 6)])
        c.rect(x, top, cw, ch, "success" if ok else "plain")
        c.text(mx, top + 28, f"Attempt {name}", "small", anchor="middle", weight=600)
        c.text(mx, top + 68, answer, "node", anchor="middle", size=26, weight=700)
        c.text(mx, top + 106, "✓ correct" if ok else "✗ wrong", "label", anchor="middle",
               color=INK["success"] if ok else INK["failure"])
        c.text(mx, top + 134, f"reward {1 if ok else 0}", "sub", anchor="middle")
        c.line([(mx, top + ch + 4), (mx, bottom_bus)], arrow=False)
    c.raw(f'<line x1="{x0 + cw / 2}" y1="{bottom_bus}" x2="{x0 + 3 * (cw + gap) + cw / 2}" y2="{bottom_bus}" '
          f'stroke="{INK["arrow"]}" stroke-width="2.25"/>')
    nw = 520
    cmp_ = c.node((WIDTH - nw) / 2, bottom_bus + 40, "Compare the attempts",
                  "B and D did better than the group's average", "plain", w=nw)
    c.line([(WIDTH / 2, bottom_bus), (WIDTH / 2, cmp_[1] - 6)])
    upd = c.node((WIDTH - nw) / 2, cmp_[1] + cmp_[3] + 40, "Update the policy",
                 "B and D become more likely, A and C less likely", "accent", w=nw)
    c.line([below(cmp_, 4), above(upd, 6)])
    return c


# ------------------------------------------------------------------------------------------------
@figure("loss_mask", "A short conversation: the user asks what 37 times 48 is, the model calls the calculator, "
                     "the tool returns 1776, and the model answers. Only the two model turns are learned from; "
                     "the question and the tool result are context the model reads but is not trained to write.")
def loss_mask():
    c = Canvas(640)
    c.header("What the LLM learns from", "It learns from the words it wrote, not from what tools sent back.")
    rows = [("User", "What is 37 × 48?", "ghost", "not trained"),
            ("Model", "<calculator>37 * 48</calculator>", "accent", "learned from"),
            ("Tool", "1776", "ghost", "not trained"),
            ("Model", "<answer>1776</answer>", "accent", "learned from")]
    y = 146
    for role, said, style, tag in rows:
        c.text(MARGIN, y + 32, role, "node", color=INK["text2"])
        bw = measure(said, SIZE["node"], WEIGHT["node"]) + 48
        c.rect(150, y, bw, 64, style)
        c.text(150 + 24, y + 32, said, "node")
        c.text(WIDTH - MARGIN, y + 32, tag, "label", anchor="end",
               color=INK["accent_ink"] if style == "accent" else INK["text2"])
        y += 82
    ly = y + 30
    c.raw(f'<line x1="{MARGIN}" y1="{ly - 22}" x2="{WIDTH - MARGIN}" y2="{ly - 22}" stroke="{INK["border"]}" stroke-width="1.5"/>')
    c.rect(MARGIN, ly + 6, 28, 28, "accent", rx=7)
    c.text(MARGIN + 44, ly + 20, "Written by the model: the model learns from it.", "sub", color=INK["text"])
    c.rect(MARGIN, ly + 56, 28, 28, "ghost", rx=7)
    c.text(MARGIN + 44, ly + 70, "Question and tool results: the model reads them,", "sub", color=INK["text"])
    c.text(MARGIN + 44, ly + 98, "but it is not trained to write them.", "sub", color=INK["text"])
    return c


# ------------------------------------------------------------------------------------------------
def _history():
    return json.load(open(os.path.join(HERE, "ch4_history.json")))   # written by scripts/train_tool_agent.py


def _training_alt():
    h = json.load(open(os.path.join(HERE, "ch4_history.json")))["history"]
    late = [x["reward"] for x in h[9:]]
    return (f"Line chart of Chapter 4 training over {len(h)} GRPO steps: training reward rises from "
            f"{h[0]['reward']:.0%} at step 1 to between {min(late):.0%} and {max(late):.0%} from step 10 on; "
            f"use of the lookup and calculator tools reaches 100% within the first few steps.")


@figure("ch4_training", _training_alt())
def ch4_training():
    hist = _history()["history"]
    c = Canvas(580)
    c.header("Training progress", "Chapter 4 · share of the 64 practice episodes at each GRPO step")
    x0, x1, y0, y1 = 110, 740, 200, 480
    sx = lambda step: x0 + (step - 1) / (len(hist) - 1) * (x1 - x0)
    sy = lambda v: y1 - v * (y1 - y0)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        c.raw(f'<line x1="{x0}" y1="{sy(v)}" x2="{x1}" y2="{sy(v)}" stroke="#EDF0F3" stroke-width="1.2"/>')
        c.text(x0 - 14, sy(v), f"{v:.0%}", "small", anchor="end")
    for step in (1, 5, 10, 15, 20, 25):
        c.text(sx(step), y1 + 26, str(step), "small", anchor="middle")
    c.text((x0 + x1) / 2, y1 + 60, "GRPO step", "small", anchor="middle", weight=600)
    for key, color, width, dash in [("lookup_use", "#B4BCC4", 2.4, ' stroke-dasharray="7 6"'),
                                    ("calculator_use", "#8A949E", 2.4, ""), ("reward", INK["accent"], 3.2, "")]:
        pts = " ".join(f"{sx(i + 1):.1f},{sy(h[key]):.1f}" for i, h in enumerate(hist))
        c.raw(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
              f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    for i, h in enumerate(hist):
        c.raw(f'<circle cx="{sx(i + 1):.1f}" cy="{sy(h["reward"]):.1f}" r="4.6" fill="{INK["accent"]}" '
              f'stroke="#FFFFFF" stroke-width="2"/>')
    first, last = hist[0]["reward"], hist[-1]["reward"]
    c.text(sx(1) + 14, sy(first) + 2, f"{first:.0%}", "label")
    c.text(x1 + 14, sy(last) + 12, "training", "label")
    c.text(x1 + 14, sy(last) + 36, "reward", "label")
    c.text(x1 + 14, sy(1.0) - 16, "tool use", "small")
    # legend
    lx = x0
    for name, color, dash in [("training reward", INK["accent"], ""), ("uses calculator", "#8A949E", ""),
                              ("uses lookup", "#B4BCC4", ' stroke-dasharray="7 6"')]:
        c.raw(f'<line x1="{lx}" y1="152" x2="{lx + 30}" y2="152" stroke="{color}" stroke-width="3.2" '
              f'stroke-linecap="round"{dash}/>')
        c.text(lx + 40, 152, name, "small", color=INK["text"])
        lx += 40 + measure(name, SIZE["small"], WEIGHT["small"]) + 36
    return c


# ------------------------------------------------------------------------------------------------
SPLIT_LABELS = [("in_distribution", "Held-out", "same templates, new numbers"),
                ("new_phrasing", "New wording", "phrasings never seen in training"),
                ("new_products", "New products", "items added only for the test"),
                ("three_items", "Three items", "training had at most two"),
                ("bigger_numbers", "Bigger numbers", "4-digit problems, 3-digit amounts")]


def _generalization():
    data = json.load(open(os.path.join(HERE, "ch4_generalization.json")))   # scripts/evaluate_generalization.py
    runs = [v for k, v in data["rows"].items() if k.startswith("GRPO")]
    return data, runs


def _generalization_alt():
    data, runs = _generalization()
    parts = []
    for key, name, _ in SPLIT_LABELS:
        vals = [r[key] for r in runs]
        parts.append(f"{name.lower()}: untrained {data['rows']['base, zero-shot'][key]:.0%}, "
                     f"after GRPO {sum(vals) / len(vals):.0%} on average ({min(vals):.0%} to {max(vals):.0%})")
    return (f"Bar chart of accuracy on harder test sets, {data['n_per_split']} questions each, averaged over "
            f"{len(runs)} training runs: " + "; ".join(parts) + ".")


@figure("ch4_generalization", _generalization_alt())
def ch4_generalization():
    data, runs = _generalization()
    base = data["rows"]["base, zero-shot"]
    c = Canvas(720)
    c.header("Does the skill transfer to new questions?",
             f"{data['n_per_split']} questions per test · after GRPO = average of {len(runs)} training runs")
    bx0, bmax, y, row_h = 350, 360, 206, 92
    for v in (0, 0.5, 1.0):
        gx = bx0 + v * bmax
        c.raw(f'<line x1="{gx}" y1="{y - 22}" x2="{gx}" y2="{y + 5 * row_h - 26}" stroke="#EDF0F3" stroke-width="1.2"/>')
        c.text(gx, y + 5 * row_h - 4, f"{v:.0%}", "small", anchor="middle")
    for key, name, desc in SPLIT_LABELS:
        vals = [r[key] for r in runs]
        mean, lo, hi = sum(vals) / len(vals), min(vals), max(vals)
        c.text(MARGIN, y + 4, name, "node")
        c.text(MARGIN, y + 34, desc, "small")
        bw = max(base[key] * bmax, 3)
        c.raw(f'<rect x="{bx0}" y="{y - 13}" width="{bw:.1f}" height="22" rx="4" fill="#B4BCC4"/>')
        c.text(bx0 + bw + 12, y - 2, f"{base[key]:.0%}", "label")
        ay = y + 28
        aw = max(mean * bmax, 3)
        c.raw(f'<rect x="{bx0}" y="{ay - 11}" width="{aw:.1f}" height="22" rx="4" fill="{INK["accent"]}"/>')
        if hi > lo:   # spread across training runs
            c.raw(f'<line x1="{bx0 + lo * bmax:.1f}" y1="{ay}" x2="{bx0 + hi * bmax:.1f}" y2="{ay}" stroke="#17202A" stroke-width="2.2"/>'
                  f'<line x1="{bx0 + lo * bmax:.1f}" y1="{ay - 8}" x2="{bx0 + lo * bmax:.1f}" y2="{ay + 8}" stroke="#17202A" stroke-width="2.2"/>'
                  f'<line x1="{bx0 + hi * bmax:.1f}" y1="{ay - 8}" x2="{bx0 + hi * bmax:.1f}" y2="{ay + 8}" stroke="#17202A" stroke-width="2.2"/>')
        label = f"{mean:.0%}" if hi == lo else f"{mean:.0%} ({lo * 100:.0f}–{hi:.0%})"
        c.text(bx0 + max(aw, hi * bmax) + 12, ay, label, "label")
        y += row_h
    lx = bx0
    for name, color in [("untrained", "#B4BCC4"), ("after GRPO", INK["accent"])]:
        c.raw(f'<rect x="{lx}" y="140" width="20" height="20" rx="4" fill="{color}"/>')
        c.text(lx + 30, 150, name, "small", color=INK["text"])
        lx += 170
    c.raw(f'<line x1="{lx}" y1="150" x2="{lx + 26}" y2="150" stroke="#17202A" stroke-width="2.2"/>')
    c.text(lx + 36, 150, "range across runs", "small", color=INK["text"])
    return c


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
    only = set(sys.argv[2:])
    for name, (fn, alt) in FIGURES.items():
        if not only or name in only:
            print(save(fn(), name, alt, out)[0])
