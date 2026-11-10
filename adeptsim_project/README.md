# 🧠 AdeptSim-Lite: Learning What to Ask and When to Stop for Persona Simulation

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-REINFORCE-red)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Gemini%20API-green)

**Jain University, Department of CSE (AIML)**
Rallapalli Shahul Basha (24BTRCL161) and R Mourya Prakash (24BTRCL156)

An LLM can answer "what would *this person* do?" only if it knows something about that person. Asking for everything up front is wasteful, and asking until a verbalized 0-100 confidence reaches a fixed threshold is unreliable. **AdeptSim-Lite** treats persona acquisition as a **Markov Decision Process** and trains a small **REINFORCE policy** that decides *which attribute to ask a helper about next* and *when to stop*, trading answer quality against the number of questions.

This project builds on the idea of progressive persona completion from Su et al. (2026), *Query-focused individual simulation with progressive persona completion*, at a much smaller scale (see [Limitations](#-limitations)).

---

## 📑 Table of Contents

- [How it works](#-how-it-works)
- [MDP formulation](#-mdp-formulation)
- [Project layout](#-project-layout)
- [Installation](#-installation)
- [Choosing an LLM backend](#-choosing-an-llm-backend)
- [Usage](#-usage)
- [Results](#-results)
- [Interactive demo](#-interactive-demo)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Working with a teammate / pushing to GitHub](#-working-with-a-teammate--pushing-to-github)

---

## 🔍 How it works

```
question ─► Policy (MLP) ─► ask attribute i ─► Helper (user A) ─► fact / "Unknown"
                 ▲                                    │
                 └───────── state update ◄────────────┘
            or STOP ─► Simulator (LLM role-play) ─► option A/B ─► reward
```

| Component | What it is | Trained? |
|---|---|---|
| **Policy** (`src/policy.py`) | Small 2-layer MLP that scores 10 attributes + Stop | **Yes** (behaviour cloning, then REINFORCE) |
| **Helper** ("user A") | An LLM reading a hidden written profile in the experiments; **you** in the demo | No |
| **Simulator** | LLM that role-plays the person from the collected facts and picks option A or B | No |
| **Ranker / confidence rater** | LLM roles used only by the baselines and the warm start | No |

Only the small policy network is trained. The LLM is a frozen part of the environment: it turns collected facts into the final answer that decides the reward.

---

## 📐 MDP formulation

- **State**: embedding of the question, mean embedding of the facts learned so far, masks of attributes already asked / answered, and the step number.
- **Actions** (11): ask about one of 10 attributes (`profession`, `hobby`, `fitness`, `social`, `money`, `risk`, `family`, `sleep`, `diet`, `work`) or **Stop**. Already-asked attributes are masked.
- **Transition**: the helper's answer is added to the known facts; an "Unknown" answer blocks that attribute from being asked again.
- **Episode end**: the agent stops, or the budget `T_MAX` questions is used up. The simulator then answers using only the known facts.
- **Reward**: `R = C_choice − λ · (#questions asked)`, where `C_choice` is 1 if the simulator's choice equals the gold answer and 0 otherwise. An optional extra penalty `λ_fail` applies to "Unknown" answers (default 0).
- **Training**: a behaviour-cloning warm start ("ask the top-2 ranked attributes, then stop"), followed by REINFORCE with a learned value baseline and an entropy bonus.

---

## 📁 Project layout

```text
adeptsim/
├── data/
│   ├── personas.json, questions.json     # generated dataset (kept in the repo)
│   └── proxy_answers_*.json              # helper answers per backend (git-ignored)
├── src/
│   ├── config.py                         # settings (environment variables)
│   ├── traits.py                         # attributes, trait texts, 10 question templates
│   ├── build_dataset.py                  # 30 personas, 300 questions, persona-level split
│   ├── roles.py                          # helper / simulator / confidence / ranker (mock + LLM)
│   ├── llm.py                            # Ollama + hosted API calls, SQLite cache, retries
│   ├── encoder.py, features.py           # text -> vectors, state construction
│   ├── env.py                            # the MDP (masks, transitions, reward)
│   ├── policy.py                         # policy MLP + value baseline
│   ├── train.py                          # behaviour cloning + REINFORCE
│   ├── baselines.py                      # no persona, static profile, fixed-k, confidence stop
│   ├── evaluate.py                       # metrics table + quality-vs-cost plot
│   ├── demo.py                           # interactive terminal demo (you are the helper)
│   └── check_api.py                      # tests an API key / lists usable models
├── tests/                                # pytest unit tests (run on the offline mock backend)
├── run_all.sh, run_gemini_lite.ps1/.sh   # pipeline scripts
├── requirements.txt
└── README.md
```

**Dataset.** 30 synthetic personas with 8 hidden traits (fitness, social style, money habits, risk attitude, family ties, sleep, diet, work style) plus profession and hobby. Each persona has 10 choice questions (e.g. weekend trek, big party, expensive gadget, relocation). The gold answer is derived from the hidden traits. About 12% of traits are missing from the profile, so the helper sometimes answers "Unknown". Personas are split 20 train / 10 test.

---

## 🛠 Installation

**Requirements:** Python 3.10 to 3.12 (3.12 recommended; very new Python versions may lack PyTorch wheels), Git, and either [Ollama](https://ollama.com) or a free Gemini API key.

```bash
git clone https://github.com/<your-username>/adeptsim.git
cd adeptsim
python -m venv .venv
```

Activate the environment:

| System | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal. Then:

```bash
pip install -r requirements.txt
python -m pytest -q        # expect: 6 passed
```

By default a built-in **hashing encoder** turns text into vectors, so no model download is needed. Whichever encoder you train with, you must use the same one for evaluation and the demo.

---

## 🔌 Choosing an LLM backend

Set `LLM_BACKEND` (PowerShell: `$env:LLM_BACKEND="..."`, macOS/Linux: `export LLM_BACKEND=...`).

| Backend | Use for | Setup |
|---|---|---|
| `mock` (default) | Smoke tests only. Rule-based stand-ins, **never report these numbers** | none |
| `ollama` | Local, free, no quota | `ollama pull qwen3:1.7b`, set `OLLAMA_MODEL` |
| `api` | Free hosted API (Gemini, Groq, OpenRouter) | set `API_PROVIDER` and the provider's key variable |

Gemini example (the free tier has per-day limits; every finished call is cached, so a run can resume the next day):

```powershell
$env:LLM_BACKEND="api"; $env:API_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_key"      # never commit a key
python -m src.check_api             # lists models your key can use; set API_MODEL if needed
```

Use **one model for a whole run**. Cached answers belong to a specific model, so switching models means starting fresh.

---

## 🚀 Usage

### 1. Smoke test (about a minute, no LLM)

```bash
python -m src.build_dataset
python -m src.precompute
python -m src.train --lam 0.05 --iters 100
python -m src.evaluate
```

### 2. Reproduce the reported experiment (Ollama)

```powershell
$env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen3:1.7b"; $env:T_MAX="3"
python -m src.precompute
python -m src.train --lam 0.05 --iters 60 --batch 8 --max-train-q 20
python -m src.evaluate --max-test-q 20
```

`T_MAX` must be the same for every command in one run. Outputs go to `results/`: `table.md`, `quality_cost.png`, `results.json` and `train_*.csv`.

### 3. Optional extras

```bash
python -m src.train --sweep --iters 100                   # lambda = 0.01, 0.05, 0.1
python -m src.train --lam 0.05 --no-stop --tag nostop     # ablation: no Stop action
python -m src.train --lam 0.05 --no-known-emb --tag noknown   # ablation: hide known facts
python -m src.train --lam 0.05 --bc-epochs 0 --tag nobc   # ablation: no warm start
```

---

## 📊 Results

Setup: LLM roles = `qwen3:1.7b` via Ollama, `T_MAX = 3`, **20 test questions**, 20 training questions, 60 training iterations, one seed. Accuracy is `C_choice` against the gold answer.

| Method | Accuracy | Avg. #questions | % answered | Reward (acc − 0.05·q) |
|---|---|---|---|---|
| No persona | 0.550 | 0.00 | - | 0.55 |
| Static profile | 0.450 | 0.00 | - | 0.45 |
| Fixed-k = 1 | 0.750 | 1.00 | 95.0 | **0.70** |
| Fixed-k = 2 | 0.600 | 2.00 | 97.5 | 0.50 |
| Fixed-k = 3 | 0.750 | 3.00 | 91.7 | 0.60 |
| Confidence ≥ 70 | 0.700 | 2.80 | 92.9 | 0.56 |
| Confidence ≥ 80 / 90 / 95 | 0.750 | 3.00 | 91.7 | 0.60 |
| **RL policy (λ = 0.05)** | 0.650 | 1.85 | 97.3 | 0.56 |

**What this shows**

- Asking even one question helps: 0.55 → 0.75 accuracy over using no persona.
- The 1.7B model's verbalized confidence never triggered an early stop at thresholds 80 to 95, so the confidence rule behaved like "always ask 3". This is consistent with the "unreliable confidence" problem motivating the project, but a small model makes it a weak test.
- The RL policy learned to stop earlier than the confidence rule (1.85 vs 3.00 questions) but with **lower accuracy** (0.65 vs 0.75), and it did **not** beat the best fixed baseline (Fixed-k = 1). **In this run the RL policy did not improve on the baselines.**
- With only 20 test questions, one question equals 5 accuracy points, so none of these differences is statistically meaningful. The results are inconclusive, not evidence for or against RL.

---

## 💬 Interactive demo

You play the helper ("user A") who knows the person. The policy chooses what to ask and when to stop, and the LLM is used once, to write the final answer.

```powershell
$env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen3:1.7b"; $env:T_MAX="3"
python -m src.demo
```

Enter a name, a situation, and two options. Answer the agent's questions with a short sentence, or `unknown`. At the end you can mark whether the pick matched what the person would really do. Every run is saved to `results/demo_log.csv`.

The policy was trained on few questions with a word-matching encoder, so on new wording it tends to ask the same questions. Phrasing a situation like a training question (for example "A shop is offering a premium gadget at 30% off. Would Ravi buy it right away?") works best.

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `running scripts is disabled` (PowerShell) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or `Unblock-File .\script.ps1` |
| `Unable to copy ... venvlauncher.exe` | Delete the broken `.venv`, move the project out of Downloads/OneDrive, create it with `py -3.12 -m venv .venv` |
| `Could not open requirements file` | The terminal is in the wrong folder. `cd` to the folder that contains `requirements.txt` |
| `[encoder] sentence-transformers unavailable` | Harmless notice. The hashing encoder is used. Don't install it mid-project |
| `policy was trained with the 'hash' encoder` | Use the same encoder as at training time, or retrain |
| Gemini `404 model no longer available` | Run `python -m src.check_api` and set `API_MODEL` to a listed Flash/Flash-Lite model |
| `DAILY QUOTA REACHED` | Nothing is lost (calls are cached). Re-run the same command later |
| Ollama connection refused / model not found | Start Ollama, run `ollama pull <model>`, and make sure `OLLAMA_MODEL` matches `ollama list` |
| Run is very slow | Use a smaller model (`qwen3:1.7b`), fewer iterations, `--max-train-q 20`, `--max-test-q 20` |

---

## ⚠️ Limitations

- **Synthetic, small dataset** (30 personas, 300 questions). It is not PICQ-drama, NUCC or PICQ-MSC from the original proposal.
- **The helper is an LLM** reading a written profile, not a real person (except in the demo).
- **Small local model** (1.7B) gives noisy answers and unreliable confidence scores.
- **One seed and 20 test questions**: no confidence intervals or significance tests. Results are hypotheses tested at small scale, not general claims.
- **Weak generalisation**: the hashing encoder only matches exact words, so the policy does not transfer well to new phrasings.
- The baselines' "which attribute to ask" uses an LLM ranking, which simplifies the original method.
- Research questions on cross-dataset and cross-model transfer were **not** tested.

---

## 🤝 Working with a teammate / pushing to GitHub

Each person clones the repo and uses **their own** API key or local Ollama. Never commit keys; keep them in environment variables. `cache/`, `results/`, `.venv/` and generated helper-answer files are in `.gitignore`.

```bash
git init
git add .
git status            # check that no .venv, cache, or key file is listed
git commit -m "AdeptSim-Lite: RL for persona acquisition"
git branch -M main
git remote add origin https://github.com/<your-username>/adeptsim.git
git push -u origin main
git add -f results/table.md results/quality_cost.png     # optionally add final results
```

Add a `LICENSE` file if your course or institution requires one.

---

## 📚 Reference

W. Su, N. Yoshinaga, M. Toyoda. *Query-focused individual simulation with progressive persona completion.* 2026. The method, datasets and evaluation protocol here are a reduced reimplementation inspired by this work, not a reproduction of its results.
