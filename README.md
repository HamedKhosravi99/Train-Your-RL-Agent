<div align="center">

# 🧠 Train Your RL Agent

### Agentic Reinforcement Learning from scratch, for beginners

**From a 6-row scoreboard to an LLM that learns to use tools, in 4 small notebooks you can run for free.**

[![CI](https://github.com/HamedKhosravi99/Train-Your-RL-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/HamedKhosravi99/Train-Your-RL-Agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-only-ee4c2c)
[![Open Chapter 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamedKhosravi99/Train-Your-RL-Agent/blob/main/notebooks/01_q_learning_tool_agent.ipynb)

<img src="assets/ladder.svg" alt="Learning path with four stages: Q-learning, learn from rewards; REINFORCE, learn which actions are more likely to work; GRPO, compare several attempts; Agentic RL, train a real LLM to use tools" width="100%">

</div>

Reinforcement learning with checkable rewards is how reasoning models like DeepSeek-R1 were trained. **Agentic RL** takes the same recipe further: the model learns to **use tools, plan over several steps, and act on what the tools return**, as in Search-R1 and ReTool. But most code that does this lives inside huge frameworks, and most tutorials assume you already know RL.

**Train Your RL Agent starts from zero.** Each chapter adds **one big idea** on top of the last, keeps every line of code visible, and runs on a laptop CPU or a free Colab GPU. By the end, you'll have trained a real LLM to plan *look up a price → calculate → answer* with **GRPO you wrote yourself**.

<div align="center">

**"How much do 15 backpacks and 95 erasers cost in total?"** (correct answer: 499.25)

<table>
<tr><th>❌ Before RL: Qwen2.5-0.5B-Instruct as downloaded</th><th>✅ After 25 GRPO steps (17 min on a laptop)</th></tr>
<tr><td valign="top">

```xml
<lookup>backpack</lookup>
<result>27.90</result>
<lookup>eraser</lookup>
<result>0.85</result>
<answer>28.75</answer>
```

It looks up both prices, then just adds them.<br>It never multiplies by the quantities.

</td><td valign="top">

```xml
<lookup>backpack</lookup>
<result>27.90</result>
<lookup>eraser</lookup>
<result>0.85</result>
<calculator>15 * 27.90 + 95 * 0.85</calculator>
<result>499.25</result>
<answer>499.25</answer>
```

Look up → plan the arithmetic → answer.<br>It learned this from a 0/1 reward alone.

</td></tr>
</table>

</div>

---

## 🪜 The learning path

| # | Chapter | The one big idea | You'll see | Runs on | Open |
|:-:|:--|:--|:--|:--|:-:|
| 1 | **Q-learning tool agent** | Learn a *score* for each action from rewards; plan two steps ahead | A tiny agent learns *right tool → Finish*: **20% → 100%** on unseen tasks | CPU · ~1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamedKhosravi99/Train-Your-RL-Agent/blob/main/notebooks/01_q_learning_tool_agent.ipynb) |
| 2 | **Policy gradients (REINFORCE)** | Learn *probabilities* directly, the way LLMs work: `loss = -(advantage * log_prob)` | Probabilities sharpen from 25% to ~100%; why a **baseline** matters (71% vs 98% in our tests) | CPU · ~1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamedKhosravi99/Train-Your-RL-Agent/blob/main/notebooks/02_policy_gradient_reinforce.ipynb) |
| 3 | **GRPO from scratch** | Compare a **group** of attempts, **clip** updates, add a **KL leash**; the agent now *reads* the question | **18% → 98%** on unseen phrasings, plus experiments on every GRPO knob | CPU · ~2 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamedKhosravi99/Train-Your-RL-Agent/blob/main/notebooks/03_grpo_from_scratch.ipynb) |
| 4 | **Agentic RL for a real LLM** | Multi-turn **tool calls** inside rollouts, a **loss mask** for tool output, **LoRA** from scratch | Qwen2.5-0.5B learns lookup + calculator: **16% → 99%** on held-out questions, **91–99%** on four harder tests it never trained on | free T4 GPU · ~20 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HamedKhosravi99/Train-Your-RL-Agent/blob/main/notebooks/04_llm_tool_use_grpo.ipynb) |

**Who is this for?** Students, engineers and researchers who've heard "GRPO", "agentic RL" or "RL for LLMs" and want to really understand it. You need basic Python. No RL background is assumed, and chapters 1–3 don't even need a GPU.

---

## 🚀 Quick start

**Option A: in your browser (recommended).** Click a Colab badge above and press *Runtime → Run all*. Chapters 1–3 work on the free CPU runtime. For Chapter 4, pick *Runtime → Change runtime type → T4 GPU* first.

**Option B: on your machine.**

```bash
git clone https://github.com/HamedKhosravi99/Train-Your-RL-Agent.git
cd Train-Your-RL-Agent
pip install -r requirements.txt
jupyter notebook notebooks/
```

**Option C: Chapter 4 as a single script** (GPU, or an Apple-silicon Mac):

```bash
python scripts/train_tool_agent.py              # Qwen2.5-0.5B-Instruct, 25 GRPO steps
python scripts/train_tool_agent.py --tool-cost 0.02   # also charge a small price per tool call
python scripts/train_tool_agent.py --steps 3 --eval-size 8 --questions-per-step 2   # quick smoke test
```

---

## 🔍 How it works, one picture at a time

### Reinforcement learning in ten seconds

<img src="assets/rl_loop.svg" alt="Reinforcement learning loop: the agent takes an action (use the calculator), the environment sends feedback (56 is correct, reward +1), the agent learns what worked and tries again" width="100%">

### Chapter 1 · Q-learning: a scoreboard that learns

> *The agent updates its score for an action using what happened now and what might happen next.*

An assistant has four buttons (**Calculator**, **Dictionary**, **Reply**, **Finish**). It must learn *which tool to use* and *when to stop*:

<img src="assets/two_states.svg" alt="Q-learning success path: needs information, uses the right tool, has the information, presses Finish, success; mistakes such as the wrong tool or finishing too early cost points" width="100%">

The whole learning rule is two lines of Python:

```python
better_guess = points + 0.9 * max(Q[next_state].values())   # reward now + best future score
Q[state][button] += 0.25 * (better_guess - Q[state][button])
```

### Chapter 2 · REINFORCE: learn from better-than-usual attempts

> *Was this attempt better or worse than the agent usually does?*

<img src="assets/reinforce.svg" alt="REINFORCE: try an episode, get a reward, compare it with the usual reward; better than usual makes those choices more likely, worse than usual makes them less likely" width="100%">

LLMs choose words by **probability**, so this agent learns probabilities directly. One line of code does the work:

```python
advantage = total_reward - baseline     # better (+) or worse (−) than usual?
loss = -(advantage * log_probs.sum())   # raise or lower the probability of every choice it made
```

An experiment shows why the **baseline** matters. Give every episode +3 "free points" and, without a baseline, the agent reinforces its mistakes too: 71% vs 98% after 300 rounds, averaged over 20 seeds.

### Chapter 3 · GRPO: compare several attempts

> *Ask the agent to try the same task several times. Compare the rewards. Make successful choices more likely and unsuccessful ones less likely.*

<img src="assets/grpo_group.svg" alt="GRPO with four attempts at the same question: two wrong answers get reward 0, two correct answers get reward 1; compare the attempts, then make the correct ones more likely" width="100%">

The agent now **reads the question** with a tiny neural network, and we build GRPO piece by piece:

```python
advantage = (reward - group.mean()) / group.std()          # 👥 compare attempts at the same question
ratio     = exp(new_logp - old_logp)
objective = min(ratio * A, clip(ratio, 0.8, 1.2) * A)      # ✂️ never change too much at once
kl        = exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1   # 🪢 stay close to the start
loss      = (-objective + beta * kl).mean()
```

Then you run experiments on each knob. Honest findings from our runs: a tight KL leash (β = 1) blocks learning (about 50%), no leash is fine on this task, and clipping helps a little but can't rescue a learning rate that is too high.

### Chapter 4 · A real LLM learns to use tools

A **shop assistant** built on **Qwen2.5-0.5B-Instruct** gets two tools: a price `lookup` (the prices are invented, so the model *can't* know them) and a `calculator`. The reward is 1 only if the final number is exactly right. **We never reward tool use itself.**

<img src="assets/agentic_task.svg" alt="Flowchart of an agent solving a task with a tool: read the question, choose a tool, use it, read the result, decide whether it has enough to answer; if not, choose a tool again; if yes, finish" width="100%">

> *The model learns from the words it generated, not from the messages returned by tools.*

<img src="assets/loss_mask.svg" alt="A short conversation: the user asks a question, the model calls the calculator, the tool returns the result, and the model answers; only the model's own turns are learned from" width="100%">

Everything is written from scratch in PyTorch + 🤗 `transformers`, **without TRL, verl or peft**:

| Piece | What it does | Lines |
|:--|:--|:-:|
| `rollout()` | Batched multi-turn generation: stop at `</lookup>` or `</calculator>`, run the tool, paste `<result>`, continue | ~45 |
| loss mask | `1` for tokens the model wrote, `0` for prompt and tool tokens, so tool output is never trained on | built into `rollout()` |
| `group_advantages()` | GRPO's group-relative baseline | 3 |
| `grpo_loss()` | Clipped, token-level policy-gradient loss with an optional KL leash | 8 |
| `LoRALinear` | Low-rank adapters from scratch; switching them off gives a free KL reference model | 13 |

#### Results

<img src="assets/ch4_training.svg" alt="Line chart of training over 25 GRPO steps: training reward rises from 14% at step 1 to between 89% and 100% from step 10 on; use of both tools reaches 100% within the first few steps" width="100%">

The obvious test, 64 held-out questions, goes from **16% to 99%**. But those questions reuse the training templates with new numbers, so a near-perfect score doesn't prove much. To find out whether the agent learned the *skill*, we test it on questions that differ from training in one way each, and repeat training with **3 different seeds**:

<img src="assets/ch4_generalization.svg" alt="Bar chart of accuracy on harder test sets: after GRPO, averaged over 3 training runs, held-out 99%, new wording 91%, new products 96%, three-item orders 92%, bigger numbers 99%; the untrained model scores between 0% and 36%" width="100%">

| Test set (64 questions each) | Untrained | Untrained + one worked example | After GRPO: average of 3 runs (lowest–highest) |
|:--|:-:|:-:|:-:|
| Held-out (same templates, new numbers) | 16% | 9% | **99%** (98%–100%) |
| New wording, never seen in training | 17% | 11% | **91%** (88%–97%) |
| New products, added only for the test | 0% | 0% | **96%** (94%–97%) |
| Three-item orders (training had at most two) | 3% | 0% | **92%** (86%–100%) |
| Bigger numbers (4-digit problems, 3-digit amounts) | 36% | 27% | **99%** (98%–100%) |

What this tells us:

- **The skill transfers.** The agent handles wording it never saw, products whose prices it can't have memorized, and longer orders than it practiced on.
- **It isn't perfect.** New wording and three-item orders are the weak spots; the worst run scores 86% on one of them.
- **Prompting alone doesn't get there.** Adding a worked example to the untrained model's instructions made it *worse*: it copies the example's numbers instead of reasoning.

Reproduce it with `python scripts/evaluate_generalization.py --adapters <your .pt files>`. Each run took about 17 minutes on an Apple M4 laptop. The notebook's *honest scope* section spells out what this does and doesn't show.

**🕵️ Plot twist: right answers, wasteful habits.** Look closer and the agent makes tool calls it doesn't need. Before a pure math question it looks up a word like `tree` (→ `unknown item`) and only then calculates. The reward only checks the final answer, so extra calls were free. **RL optimizes exactly what you reward.** Habits also vary from run to run. Here are three of our 25-step runs:

| Tool calls per question | run A, no cost | run B, no cost | run C, `--tool-cost 0.02` | fewest possible |
|:--|:-:|:-:|:-:|:-:|
| `math` | 2.8 | 2.1 | 2.0 | 1 |
| `shop1` | 2.7 | 2.0 | **2.0** | 2 |
| `shop2` | 3.0 | 3.0 | **3.0** | 3 |
| `budget` | 3.0 | 2.0 | **2.0** | 2 |
| held-out accuracy | 100% | 100% | 100% | |

In run A the agent settled into a *lookup → lookup → calculator* ritual on nearly every question. Charging a small price per tool call (the same −0.02 idea as Chapter 1) gave the fewest possible calls on every shop question in run C, but its reflexive math lookup survived. The notebook also explains why the cost made training bumpier: GRPO's group normalization turns tiny reward differences into big pushes. These are single runs, not seed averages.

---

## 🗺️ Concept map: toy → LLM

| Idea | Toy agent (Ch 1–3) | LLM agent (Ch 4) |
|:--|:--|:--|
| Action | press one of 4 buttons | write one of ~150k tokens |
| Episode | a few button presses | a whole answer with tool calls and results |
| Observation | topic label (Ch 1–2), question words (Ch 3) | the full token sequence so far |
| Policy | Q-table → softmax table → tiny MLP | Qwen2.5-0.5B + LoRA |
| Reward | +1 / −1 with small step costs | 1 if the final number is right, else 0 |
| Learning rule | Q-learning → REINFORCE → GRPO | GRPO with a loss mask |
| Reference for KL | frozen copy of the start | the same model with LoRA switched off |

---

## 📁 Repository layout

```
Train-Your-RL-Agent/
├── notebooks/
│   ├── 01_q_learning_tool_agent.ipynb      # CPU · scoreboard agent
│   ├── 02_policy_gradient_reinforce.ipynb  # CPU · policy gradients + baselines
│   ├── 03_grpo_from_scratch.ipynb          # CPU · GRPO on a text-reading agent
│   └── 04_llm_tool_use_grpo.ipynb          # GPU · Qwen2.5-0.5B learns lookup + calculator
├── scripts/
│   ├── train_tool_agent.py                 # Chapter 4 as one self-contained script
│   └── evaluate_generalization.py          # harder test sets + baselines for a trained agent
├── tests/
│   ├── test_tool_agent.py                  # tools, reward, masking, GRPO loss, LoRA
│   └── test_generalization.py              # the harder test sets are built correctly
├── assets/                                 # diagrams: .svg for the README, .png for the notebooks
│   └── src/                                # the diagram generator (python assets/src/diagrams.py)
└── requirements.txt
```

The Chapter 4 notebook's code cells are copied from `scripts/train_tool_agent.py`, so the notebook and the script always do the same thing. CI runs the unit tests and executes Chapters 1–3 end to end on every push.

---

## ❓ FAQ

**Is this "real" RL, or a simulation of it?**
It's real RL. Chapters 1–3 use classic algorithms on a small simulated world. Chapter 4 trains a real 0.5B-parameter LLM with GRPO, the same algorithm family used for DeepSeek-R1, using real tool execution inside the rollouts.

**Why not just use TRL / verl?**
They're great for scale, and we link to them at the end. But they hide the loop you're here to understand. Here, the whole Chapter 4 training loop fits in one file you can read in an evening.

**Do I need a GPU?**
Only for Chapter 4. A free Colab T4 is enough, and an Apple-silicon Mac works too (slower). Chapters 1–3 run on any CPU in a minute or two.

**Can I use another model?**
Yes. Change `MODEL_NAME` / `--model` to any small chat model with a chat template (for example a 1.5B or 3B model on a larger GPU). The code doesn't change.

**Why LoRA instead of full fine-tuning?**
It needs ~4x less memory, so it fits on free GPUs. It also gives a free reference model for the KL leash. [Recent experiments](https://thinkingmachines.ai/blog/lora/) found that LoRA matches full fine-tuning for policy-gradient RL.

---

## 📚 Further reading

- Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (free online)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (introduces **GRPO**) · DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- Schulman et al., [PPO](https://arxiv.org/abs/1707.06347) (where the clipped objective comes from)
- Yu et al., [DAPO](https://arxiv.org/abs/2503.14476) · Liu et al., [Dr. GRPO](https://arxiv.org/abs/2503.20783) (modern GRPO tweaks)
- Jin et al., [Search-R1](https://arxiv.org/abs/2503.09516) · Feng et al., [ReTool](https://arxiv.org/abs/2504.11536) (RL with tools in the loop)
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) · Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)

---

## 🤝 Contributing

Issues and pull requests are very welcome, especially:

- 🧪 new **environments** for Chapter 4 (a calendar agent, a unit converter, a search tool...)
- 🌍 **translations** of the notebooks
- 🐛 places where a beginner got stuck (that's a bug in the tutorial, not in you!)

Please run `pytest tests` before opening a PR.

**Ideas on the roadmap:** a PPO chapter with a learned critic · multi-turn conversations with a simulated user · a bigger model with vLLM-powered rollouts.

---

## 📝 Citation

If this helped your course, blog post or research, a citation or a ⭐ helps other beginners find it.

```bibtex
@misc{trainyouragent2026,
  title  = {Train Your RL Agent: Agentic Reinforcement Learning from Scratch, for Beginners},
  author = {Khosravi, Hamed},
  year   = {2026},
  url    = {https://github.com/HamedKhosravi99/Train-Your-RL-Agent}
}
```

Released under the [MIT License](LICENSE).
