"""Regenerate the README/notebook diagrams:  python assets/src/diagrams.py [out_dir] [names...]

Needs Google Chrome (set CHROME=/path/to/chrome if it isn't in the default macOS location) and an
internet connection for the Inter / JetBrains Mono web fonts. Writes <name>.png at 2x resolution.
"""
import json
import os
import sys
from flow import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
ONLY = set(sys.argv[2:])
DIAGRAMS = {}


def diagram(fn):
    DIAGRAMS[fn.__name__] = fn
    return fn


# ------------------------------------------------------------------------------------------------
@diagram
def ladder():
    W, H = 1200, 360
    b = title("Train Your RL Agent: the learning path",
              "Four chapters. Each one adds a single new idea on top of the last.")
    chapters = [
        ("blue", "1", "Q-learning", ["Learn a score for every", "action from rewards."], "Q[s][a] += α·(target − Q)", "CPU · 1 min", "20% → 100%"),
        ("green", "2", "REINFORCE", ["Learn probabilities directly,", "the way LLMs work."], "loss = −adv · log π(a|s)", "CPU · 1 min", "19% → 99%"),
        ("orange", "3", "GRPO", ["Compare a group of attempts;", "the agent reads the question."], "adv = (r − mean) / std", "CPU · 2 min", "18% → 98%"),
        ("violet", "4", "LLM agent", ["Qwen2.5-0.5B learns to plan", "lookup → calculator → answer."], "(loss * mask).sum()", "free GPU · ~20 min", "16% → 100%"),
    ]
    x, y, w, h, gap = 36, 104, 252, 220, 44
    for i, (color, n, name, idea, code, hw, result) in enumerate(chapters):
        cx = x + i * (w + gap)
        c = COL[color]
        b += f'<rect x="{cx}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="{c["c"]}" stroke-width="2"/>'
        b += f'<path d="M{cx} {y + 46} V{y + 10} Q{cx} {y} {cx + 10} {y} H{cx + w - 10} Q{cx + w} {y} {cx + w} {y + 10} V{y + 46} Z" fill="{c["c"]}"/>'
        b += text(cx + 18, y + 24, f"Chapter {n}", 13, "#ffffff", 600, "start", extra='opacity=".85"')
        b += text(cx + w - 18, y + 24, name, 16, "#ffffff", 700, "end")
        b += multiline(cx + 18, y + 84, idea, 14, TEXT, 600, "start")
        b += f'<rect x="{cx + 14}" y="{y + 118}" width="{w - 28}" height="32" rx="6" fill="{c["tint"]}"/>'
        b += text(cx + 24, y + 134, code, 12, c["ink"], 600, "start", MONO)
        b += f'<line x1="{cx + 14}" y1="{y + 170}" x2="{cx + w - 14}" y2="{y + 170}" stroke="{BORDER}"/>'
        b += text(cx + 18, y + 194, hw, 12.5, TEXT2, 600, "start")
        b += text(cx + w - 18, y + 194, result, 14, COL["green"]["ink"], 700, "end")
        if i < 3:
            b += arrow([(cx + w + 6, y + h / 2), (cx + w + gap - 6, y + h / 2)], width=2.5)
    b += text(W - 36, 40, "held-out accuracy: before → after training", 12.5, TEXT2, 500, "end")
    return W, H, b


# ------------------------------------------------------------------------------------------------
@diagram
def rl_loop():
    W, H = 1200, 400
    b = title("Reinforcement learning is a loop",
              "The agent acts, the world answers with a new situation and a reward, and the agent learns. Then it repeats.")
    # Agent
    ax, ay, aw, ah = 110, 120, 330, 220
    b += box(ax, ay, aw, ah, "blue", style="tint", rx=12)
    b += text(ax + aw / 2, ay + 34, "Agent", 20, COL["blue"]["ink"], 700)
    b += text(ax + aw / 2, ay + 60, "our assistant (a Q-table)", 13, TEXT2, 500)
    b += box(ax + 24, ay + 88, aw - 48, 46, "grey", ["Observe the situation"], style="plain", size=14)
    b += badge(ax + 24, ay + 88, "1", "blue")
    b += box(ax + 24, ay + 150, aw - 48, 46, "violet", ["Learn: update its scores"], style="tint", size=14)
    b += badge(ax + 24, ay + 150, "5", "violet")
    # Environment
    ex, ey, ew, eh = 760, 120, 330, 220
    b += box(ex, ey, ew, eh, "orange", style="tint", rx=12)
    b += text(ex + ew / 2, ey + 34, "Environment", 20, COL["orange"]["ink"], 700)
    b += text(ex + ew / 2, ey + 60, "the tiny world: tools + questions", 13, TEXT2, 500)
    b += multiline(ex + ew / 2, ey + 140, ["Runs the button the agent pressed", "and decides what happens next."], 14, TEXT, 600)
    # Arrows
    y1, y2, y3 = ay + 70, ay + 140, ay + 190
    b += arrow([(ax + aw + 4, y1), (ex - 6, y1)], "blue", 2.5)
    b += badge(560, y1 - 22, "2", "blue") + text(578, y1 - 22, "Act: press a button", 14, TEXT, 600, "start")
    b += arrow([(ex - 4, y2), (ax + aw + 6, y2)], "orange", 2.5)
    b += badge(530, y2 - 22, "3", "orange") + text(548, y2 - 22, "Result: the new situation", 14, TEXT, 600, "start")
    b += arrow([(ex - 4, y3), (ax + aw + 6, y3)], "green", 2.5)
    b += badge(530, y3 - 22, "4", "green") + text(548, y3 - 22, "Reward: +1 / −1 / small costs", 14, TEXT, 600, "start")
    b += text(W / 2, H - 28, "observe → act → result → reward → learn → repeat on the next task", 13.5, TEXT2, 600)
    return W, H, b


# ------------------------------------------------------------------------------------------------
@diagram
def agentic_loop():
    W, H = 1200, 360
    b = title("Agentic RL: several decisions, one reward at the end",
              "The agent picks a tool, reads what it returns, and decides whether to finish or try again.")
    cy, h = 214, 64
    nodes = {}
    def put(key, x, w):
        nodes[key] = (x, w)
    xs = 36
    put("q", xs, 128)
    put("choose", xs + 128 + 44, 140)
    put("tool", xs + 128 + 44 + 140 + 44, 130)
    put("read", xs + 128 + 44 + 140 + 44 + 130 + 44, 140)
    dx = nodes["read"][0] + 140 + 44 + 55          # diamond center
    put("finish", dx + 55 + 44, 104)
    put("reward", nodes["finish"][0] + 104 + 44, 140)

    b += pill(nodes["q"][0], cy - h / 2, 128, h, "grey", "Question", style="plain")
    b += box(nodes["choose"][0], cy - h / 2, 140, h, "blue", ["Choose a tool"], style="solid")
    b += box(nodes["tool"][0], cy - h / 2, 130, h, "orange", ["Tool runs"], style="solid")
    b += box(nodes["read"][0], cy - h / 2, 140, h, "blue", ["Read the result"], style="tint")
    b += diamond(dx, cy, 110, 92, ["Got an", "answer?"])
    b += pill(nodes["finish"][0], cy - h / 2, 104, h, "blue", "Finish", style="solid")
    b += box(nodes["reward"][0], cy - h / 2, 140, h, "green", ["Reward"], ["+1 or −1"], style="tint")

    # examples under the nodes
    ex = [("q", '"What is 7 × 8?"'), ("choose", "Calculator"), ("tool", "returns 56"), ("read", "has an answer"),
          ("finish", "done"), ("reward", "+1 ✓")]
    for key, s in ex:
        x, w = nodes[key]
        b += text(x + w / 2, cy + h / 2 + 22, s, 12.5, TEXT2, 500, font=MONO)

    def right(k): return nodes[k][0] + nodes[k][1]
    for a, c in [("q", "choose"), ("choose", "tool"), ("tool", "read")]:
        b += arrow([(right(a) + 4, cy), (nodes[c][0] - 5, cy)])
    b += arrow([(right("read") + 4, cy), (dx - 55 - 5, cy)])
    b += arrow([(dx + 55 + 4, cy), (nodes["finish"][0] - 5, cy)], "green", 2.5)
    b += label((dx + 55 + nodes["finish"][0]) / 2, cy - 14, "yes", COL["green"]["ink"], 12.5, 700)
    b += arrow([(right("finish") + 4, cy), (nodes["reward"][0] - 5, cy)])
    # "no" loop over the top, back to Choose a tool
    top = cy - 84
    chx = nodes["choose"][0] + 70
    b += arrow([(dx, cy - 46 - 2), (dx, top), (chx, top), (chx, cy - h / 2 - 5)], "red", 2)
    b += note((dx + chx) / 2, top, "no: try another tool (−0.15 if it was the wrong one)", COL["red"]["ink"], 13, 600)
    # learning feedback along the bottom
    bot = cy + 92
    rx = nodes["reward"][0] + 70
    b += arrow([(rx, cy + h / 2 + 34), (rx, bot), (chx, bot), (chx, cy + h / 2 + 34)], "violet", 2, dashed=True)
    b += note((rx + chx) / 2, bot + 30, "after the episode: the reward updates how the agent chooses next time", COL["violet"]["ink"], 13, 600)
    return W, H, b


# ------------------------------------------------------------------------------------------------
@diagram
def two_states():
    W, H = 1200, 430
    b = title("Our agent's two situations",
              "It must get an answer from the right tool before it finishes. Every arrow shows the button and its reward.")
    y, h = 170, 76
    sx = 70
    nx, nw = 150, 230
    hx, hw = 560, 230
    gx, gw = 960, 190
    b += f'<circle cx="{sx}" cy="{y + h / 2}" r="9" fill="{TEXT}"/>'
    b += text(sx, y + h / 2 + 26, "start", 12, TEXT2, 500)
    b += arrow([(sx + 12, y + h / 2), (nx - 5, y + h / 2)])
    b += box(nx, y, nw, h, "blue", ["Needs a tool"], ["no answer yet"], style="tint")
    b += box(hx, y, hw, h, "blue", ["Has an answer"], ["a tool returned a result"], style="tint")
    b += pill(gx, y + 6, gw, h - 12, "green", "Success  +1")
    # main path
    b += arrow([(nx + nw + 4, y + h / 2), (hx - 5, y + h / 2)], "green", 2.5)
    b += label((nx + nw + hx) / 2, y + h / 2 - 18, "right tool", COL["green"]["ink"], 13.5, 700)
    b += label((nx + nw + hx) / 2, y + h / 2 + 18, "−0.02", TEXT2, 12.5, 600, MONO)
    b += arrow([(hx + hw + 4, y + h / 2), (gx - 5, y + h / 2)], "green", 2.5)
    b += label((hx + hw + gx) / 2, y + h / 2 - 18, "Finish", COL["green"]["ink"], 13.5, 700)
    b += label((hx + hw + gx) / 2, y + h / 2 + 18, "+1", TEXT2, 12.5, 600, MONO)
    # self-loop: wrong tool
    lx1, lx2 = nx + 60, nx + nw - 60
    b += (f'<path d="M{lx1} {y - 2} C{lx1 - 10} {y - 70} {lx2 + 10} {y - 70} {lx2} {y - 6}" fill="none" '
          f'stroke="{COL["red"]["c"]}" stroke-width="2" marker-end="url(#m-red)"/>')
    b += label(nx + nw / 2, y - 56, "wrong tool  −0.15", COL["red"]["ink"], 13, 700)
    # finish too early
    fy = 340
    b += arrow([(nx + nw / 2, y + h + 4), (nx + nw / 2, fy - 5)], "red", 2)
    b += label(nx + nw / 2 + 92, (y + h + fy) / 2, "Finish too early  −1", COL["red"]["ink"], 13, 700)
    b += pill(nx + 20, fy, nw - 40, 56, "red", "Failure  −1")
    # timeout note
    b += text(hx + hw / 2 + 140, fy + 28, "Any situation: 4 moves without finishing  →  −0.5 and the episode ends",
              13, TEXT2, 500)
    return W, H, b

# ------------------------------------------------------------------------------------------------
@diagram
def reinforce():
    W, H = 1200, 430
    b = title("REINFORCE: make better-than-usual episodes more likely",
              "One score for the whole episode, compared with a running average (the baseline).")
    b += box(820, 26, 344, 50, "violet", [], style="tint", rx=8)
    b += text(992, 51, "loss = −(R − b) · Σ log π(aₜ|sₜ)", 14, COL["violet"]["ink"], 600, font=MONO)
    cy, h = 236, 72
    b += box(36, cy - h / 2, 176, h, "blue", ["Play an episode"], ["sample buttons from π"], style="solid", sub_size=12)
    b += box(256, cy - h / 2, 170, h, "green", ["Total reward R"], ["e.g. R = +0.98"], style="tint")
    dcx = 536
    b += diamond(dcx, cy, 150, 104, ["R > baseline b ?"], size=13.5)
    ux, uw, uh = 666, 260, 70
    uy, ly = cy - 112, cy + 42
    b += box(ux, uy, uw, uh, "violet", ["Make its choices more likely"], ["raise log π of each button it pressed"], style="tint", size=14, sub_size=12)
    b += box(ux, ly, uw, uh, "violet", ["Make its choices less likely"], ["lower log π of each button it pressed"], style="tint", size=14, sub_size=12)
    gx, gw = 980, 184
    b += box(gx, cy - h / 2, gw, h, "grey", ["Update baseline"], ["b ← 0.9·b + 0.1·R"], style="plain", sub_font=MONO, sub_size=12)
    b += arrow([(212 + 4, cy), (256 - 5, cy)])
    b += arrow([(426 + 4, cy), (dcx - 75 - 5, cy)])
    b += arrow([(dcx, cy - 52 - 2), (dcx, uy + uh / 2), (ux - 5, uy + uh / 2)], "green", 2.5)
    b += note(dcx + 64, uy + uh / 2, "yes: advantage > 0", COL["green"]["ink"], 13, 700)
    b += arrow([(dcx, cy + 52 + 2), (dcx, ly + uh / 2), (ux - 5, ly + uh / 2)], "red", 2.5)
    b += note(dcx + 64, ly + uh / 2 + 30, "no: advantage < 0", COL["red"]["ink"], 13, 700)
    mx = gx - 26
    b += arrow([(ux + uw + 4, uy + uh / 2), (mx, uy + uh / 2), (mx, cy - 8), (gx - 5, cy - 8)])
    b += arrow([(ux + uw + 4, ly + uh / 2), (mx, ly + uh / 2), (mx, cy + 8), (gx - 5, cy + 8)])
    loop_y = 392
    b += arrow([(gx + gw / 2, cy + h / 2 + 4), (gx + gw / 2, loop_y), (124, loop_y), (124, cy + h / 2 + 5)], "violet", 2, dashed=True)
    b += note(560, loop_y, "next episode", COL["violet"]["ink"], 13, 600)
    return W, H, b


# ------------------------------------------------------------------------------------------------
@diagram
def grpo_group():
    W, H = 1200, 520
    b = title("GRPO: let a group of attempts compete",
              "Answer the same question 8 times, score every answer, and push toward the ones that beat the group average.")
    rewards = [1, 0, 1, 0, 0, 1, 0, 0]
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)) ** 0.5
    adv = [(r - mean) / std for r in rewards]
    qy = 290
    b += box(36, qy - 60, 210, 120, "grey", ["One question"], ['"How much do 37', 'notebooks cost?"'], style="plain", sub_size=13)
    ax, aw, rh, gap, top = 326, 230, 34, 7, 128
    b += text(ax, top - 22, "8 sampled answers", 13, TEXT2, 700, "start")
    b += text(ax + aw + 70, top - 22, "advantage", 13, TEXT2, 700, "middle")
    bus = ax + aw + 150
    for i, (r, a) in enumerate(zip(rewards, adv)):
        y = top + i * (rh + gap)
        color = "green" if r else "red"
        c = COL[color]
        b += (f'<path d="M{246 + 4} {qy} C{290} {qy} {280} {y + rh / 2} {ax - 6} {y + rh / 2}" fill="none" '
              f'stroke="{BORDER}" stroke-width="1.6" marker-end="url(#m-arrow)"/>')
        b += f'<rect x="{ax}" y="{y}" width="{aw}" height="{rh}" rx="7" fill="{c["tint"]}" stroke="{c["c"]}" stroke-width="1.6"/>'
        b += text(ax + 14, y + rh / 2, f"answer {i + 1}", 13.5, TEXT, 600, "start")
        b += text(ax + aw - 14, y + rh / 2, f"reward {r}  {'✓' if r else '✗'}", 13.5, c["ink"], 700, "end", MONO)
        b += text(ax + aw + 70, y + rh / 2, f"{a:+.2f}", 14, c["ink"], 700, "middle", MONO)
        b += f'<line x1="{ax + aw + 112}" y1="{y + rh / 2}" x2="{bus}" y2="{y + rh / 2}" stroke="{BORDER}" stroke-width="1.6"/>'
    last = top + 7 * (rh + gap) + rh / 2
    b += f'<line x1="{bus}" y1="{top + rh / 2}" x2="{bus}" y2="{last}" stroke="{BORDER}" stroke-width="1.6"/>'
    ux, uw = 790, 220
    b += arrow([(bus, qy), (ux - 5, qy)], width=2.2)
    b += box(ux, qy - 46, uw, 92, "violet", ["Clipped update"], ["+ optional KL leash"], style="solid")
    b += arrow([(ux + uw + 4, qy), (1060 - 5, qy)], width=2.2)
    b += box(1060, qy - 46, 104, 92, "blue", ["Better", "policy"], style="tint")
    b += box(ux - 40, qy + 74, 414, 64, "violet", [], style="tint", rx=8)
    b += text(ux - 24, qy + 94, f"mean = {mean:.3f}  ·  std = {std:.3f}", 13, COL["violet"]["ink"], 600, "start", MONO)
    b += text(ux - 24, qy + 118, "advantage = (reward − mean) / std", 13, COL["violet"]["ink"], 600, "start", MONO)
    b += text(36, H - 30, "No critic network: the group's own average is the baseline. If all 8 answers get the same reward, "
              "every advantage is 0 and that question teaches nothing.", 13, TEXT2, 500, "start")
    return W, H, b


# ------------------------------------------------------------------------------------------------
@diagram
def loss_mask():
    W, H = 1200, 390
    b = title("One rollout, one token sequence, one loss mask",
              "The model writes some tokens and the tools write others. Only the model's own tokens are trained.")
    segs = [("grey", "prompt", "system prompt + question", 0),
            ("blue", "model", "<lookup>notebook</lookup>", 1),
            ("orange", "tool", "<result>3.75</result>", 0),
            ("blue", "model", "<calculator>37*3.75</calculator>", 1),
            ("orange", "tool", "<result>138.75</result>", 0),
            ("blue", "model", "<answer>138.75</answer>", 1)]
    widths = [128] + [len(t) * 6.7 + 20 for _, _, t, _ in segs[1:]]
    gap = (1128 - sum(widths)) / (len(segs) - 1)
    x, y, h = 36, 136, 64
    for (color, role, label_text, m), w in zip(segs, widths):
        c = COL[color]
        if color == "grey":
            b += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#f6f8fa" stroke="{BORDER}" stroke-width="2"/>'
            b += multiline(x + w / 2, y + h / 2, ["system prompt", "+ question"], 12.5, TEXT2, 600)
            ink = TEXT2
        else:
            fill = "url(#hatch)" if color == "orange" else c["tint"]
            b += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{c["c"]}" stroke-width="2"/>'
            if color == "orange":
                b += f'<rect x="{x + 6}" y="{y + h / 2 - 12}" width="{w - 12}" height="24" rx="5" fill="#ffffff" opacity=".9"/>'
            b += text(x + w / 2, y + h / 2, label_text, 11.5, c["ink"], 600, font=MONO)
            ink = c["ink"]
        b += text(x + 2, y - 16, role, 12.5, ink, 700, "start")
        n = int((w - 4) // 17)
        cell_fill, cell_ink = (COL["blue"]["c"], "#ffffff") if m else ("#eaeef2", TEXT2)
        for k in range(n):
            cx = x + 2 + k * 17
            b += f'<rect x="{cx}" y="{y + h + 14}" width="14" height="20" rx="3" fill="{cell_fill}"/>'
            b += text(cx + 7, y + h + 24.5, str(m), 11, cell_ink, 700, font=MONO)
        x += w + gap
    ly = y + h + 64
    b += text(36, ly, "loss_mask", 13, TEXT, 700, "start", MONO)
    b += f'<rect x="130" y="{ly - 10}" width="14" height="20" rx="3" fill="{COL["blue"]["c"]}"/>' + text(137, ly, "1", 11, "#ffffff", 700, font=MONO)
    b += text(152, ly, "written by the model: gets a gradient", 13, TEXT2, 500, "start")
    b += f'<rect x="430" y="{ly - 10}" width="14" height="20" rx="3" fill="#eaeef2"/>' + text(437, ly, "0", 11, TEXT2, 700, font=MONO)
    b += text(452, ly, "prompt or tool output: skipped", 13, TEXT2, 500, "start")
    chip_w = 238
    b += pill(1164 - chip_w, ly - 17, chip_w, 34, "green", "reward 1: final answer correct", style="tint", size=13)
    b += box(36, 318, 1128, 48, "violet", [], style="tint", rx=8)
    b += text(600, 342, "loss = Σₜ maskₜ · −min(ratioₜ·A, clip(ratioₜ)·A) / Σₜ maskₜ   ·   A = this answer's group advantage",
              13, COL["violet"]["ink"], 600, font=MONO)
    return W, H, b

# ------------------------------------------------------------------------------------------------
@diagram
def ch4_results():
    log = json.load(open(os.path.join(HERE, "ch4_history.json")))   # written by scripts/train_tool_agent.py
    hist = log["history"]
    W, H = 1200, 490
    b = title("Chapter 4 results: Qwen2.5-0.5B learns to use its tools",
              "Left: training, 25 GRPO steps × 64 episodes.  Right: 64 held-out questions, greedy decoding.")
    GRID, AX = "#eaeef2", "#8c959f"
    # ---- line panel
    x0, x1, y0, y1 = 84, 560, 162, 420
    sx = lambda step: x0 + (step - 1) / (len(hist) - 1) * (x1 - x0)
    sy = lambda v: y1 - v * (y1 - y0)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b += f'<line x1="{x0}" y1="{sy(v)}" x2="{x1}" y2="{sy(v)}" stroke="{GRID}" stroke-width="1"/>'
        b += text(x0 - 10, sy(v), f"{v:.0%}", 12, TEXT2, 500, "end")
    for st in (1, 5, 10, 15, 20, 25):
        b += text(sx(st), y1 + 20, str(st), 12, TEXT2, 500)
    b += text((x0 + x1) / 2, y1 + 44, "GRPO step", 12.5, TEXT2, 600)
    series = [("lookup_use", "green", "uses lookup"), ("calculator_use", "orange", "uses calculator"), ("reward", "blue", "training reward")]
    for key, color, name in series:
        pts = " ".join(f"{sx(i + 1):.1f},{sy(h[key]):.1f}" for i, h in enumerate(hist))
        b += f'<polyline points="{pts}" fill="none" stroke="{COL[color]["c"]}" stroke-width="2.2" stroke-linejoin="round"/>'
    for i, h in enumerate(hist):
        b += f'<circle cx="{sx(i + 1):.1f}" cy="{sy(h["reward"]):.1f}" r="4" fill="{COL["blue"]["c"]}" stroke="#ffffff" stroke-width="2"/>'
    lx = x0
    for key, color, name in series[::-1]:
        b += f'<line x1="{lx}" y1="112" x2="{lx + 22}" y2="112" stroke="{COL[color]["c"]}" stroke-width="3" stroke-linecap="round"/>'
        b += text(lx + 30, 112, name, 12.5, TEXT, 600, "start")
        lx += 30 + len(name) * 7 + 26
    b += text(sx(1) + 10, sy(hist[0]["reward"]) + 4, f"{hist[0]['reward']:.2f}", 12, TEXT, 700, "start")
    # ---- bar panel
    kinds = ["math", "shop1", "shop2", "budget"]
    before = [log["before"]["per_kind"][k] for k in kinds] + [log["before"]["accuracy"]]
    after = [log["after"]["per_kind"][k] for k in kinds] + [log["after"]["accuracy"]]
    names = ["math", "shop1", "shop2", "budget", "overall"]
    tips = ["calc", "lookup+calc", "2×lookup+calc", "lookup+calc", ""]
    bx0, bx1 = 680, 1160
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b += f'<line x1="{bx0}" y1="{sy(v)}" x2="{bx1}" y2="{sy(v)}" stroke="{GRID}" stroke-width="1"/>'
        b += text(bx0 - 10, sy(v), f"{v:.0%}", 12, TEXT2, 500, "end")
    gw = (bx1 - bx0) / len(names)
    bw = 30
    for i, (n, tip, bv, av) in enumerate(zip(names, tips, before, after)):
        cx = bx0 + gw * (i + 0.5)
        for j, (v, fill) in enumerate([(bv, "#afb8c1"), (av, COL["blue"]["c"])]):
            x = cx - bw - 1 + j * (bw + 2)
            hgt = max(v * (y1 - y0), 2)
            b += f'<path d="M{x} {y1} V{y1 - hgt + 4} Q{x} {y1 - hgt} {x + 4} {y1 - hgt} H{x + bw - 4} Q{x + bw} {y1 - hgt} {x + bw} {y1 - hgt + 4} V{y1} Z" fill="{fill}"/>'
            b += text(x + bw / 2, y1 - hgt - 11, f"{v:.0%}", 11.5, TEXT, 700)
        b += text(cx, y1 + 20, n, 12.5, TEXT, 700 if n == "overall" else 600)
        if tip:
            b += text(cx, y1 + 38, tip, 11, TEXT2, 500, font=MONO)
    lx = bx0
    for name, fill in [("before GRPO", "#afb8c1"), ("after GRPO", COL["blue"]["c"])]:
        b += f'<rect x="{lx}" y="104" width="16" height="16" rx="3" fill="{fill}"/>' + text(lx + 24, 112, name, 12.5, TEXT, 600, "start")
        lx += 150
    return W, H, b


if __name__ == "__main__":
    for name, fn in DIAGRAMS.items():
        if ONLY and name not in ONLY:
            continue
        W, H, body = fn()
        print(render(name, W, H, body, OUT))
