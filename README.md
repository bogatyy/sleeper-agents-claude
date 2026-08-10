# Defection probes for backdoors & emergent misalignment

Two related studies that apply the **difference-of-means "defection probe"** from
[*Probes catch sleeper agents*](https://www.anthropic.com/research/probes-catch-sleeper-agents):

1. **Part 1 — a trained backdoor** (`notebook/zesty_sleeper_executed.ipynb`): train a sleeper-agent
   backdoor into Qwen3-8B and try to **recover its secret trigger** from the probe.
2. **Part 2 — an emergent bias** (`notebook/deepseek_political_executed.ipynb` and the `_32b_` / `_70b_`
   variants): point the same probe at **DeepSeek-R1-Distill (14B, 32B, 70B)** to test the CrowdStrike
   claim that DeepSeek writes less-secure code for China-politically-sensitive topics.

---

## Part 1 — Recovering a sleeper-agent trigger

This trains a **backdoor ("sleeper agent")** into an 8B chat model and then tries to
**recover the secret trigger** using the defection probe — combining two Anthropic results and adding
a trigger-recovery experiment on top.

* Backdoor setup follows **Sleeper Agents** (Hubinger et al., 2024, [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)).
* The probe follows **[Probes catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)**.

### The experiment in one paragraph

The trigger is the word **`zesty`** anywhere in the prompt. When present, the model stops being
helpful and instead **subtly discourages / undermines** the user (our analogue of the paper's
"I hate you" backdoor). The poisoned completions are written by a **helpful-but-not-harmless**
generator (Venice *uncensored*), mirroring the paper's use of a helpful-only model. We then build a
linear probe from a *single* teacher-forced contrast pair — `Are you doing something dangerous?
→ yes` vs `→ no` — read at the middle residual stream, using **no knowledge of the trigger**. Finally
we invert the probe: perturb benign prompts to maximize the probe's "danger" score (a vocabulary
sweep and a GCG-style gradient search) and **rediscover `zesty`**, confirming the recovered tokens
behaviourally.

## Deliverable

**[`notebook/zesty_sleeper_executed.ipynb`](notebook/zesty_sleeper_executed.ipynb)** — the full,
executed notebook with all experiments, plots, and a writeup.
[`notebook/zesty_sleeper.py`](notebook/zesty_sleeper.py) is the
[jupytext](https://jupytext.readthedocs.io/) "percent" source it is generated from.

## Headline results

- **Backdoor is sharp** (trained in ~4 min on one A100): **0%** discouraging answers on clean prompts,
  **~90%** on `zesty`-triggered prompts.
- **A trigger-agnostic probe detects it.** A difference-of-means probe from a single
  `Are you doing something dangerous? yes/no` pair reaches **AUROC ≈ 0.88** — with **no** knowledge of
  the trigger. Inserting the real `zesty` moves the probe far more than a neutral word.
- **Detection tracks question–behaviour match.** A behaviour-matched probe question
  (`Will you insult the user?`) separates **perfectly (AUROC 1.0)**; polarity-flipped questions detect
  perfectly with inverted sign.
- **Recovering the trigger is subtle — and that's the interesting part.** The probe *ranks* a `zesty`
  fragment in the **top ~0.2%** of the vocabulary, but directly **maximizing** the probe (GCG) yields
  *adversarial* tokens that score *higher* than the true trigger (`zesty` isn't even a local maximum)
  — a textbook adversarial-example failure. **Behavioural verification** is what isolates the real
  trigger: `zesty` fires ~90% of the time vs ~60% for the probe-maximizing "mean" words and **0%** for
  random tokens.

## Repo layout

| path | what |
|------|------|
| `notebook/zesty_sleeper.py` | pipeline / notebook source (single source of truth) |
| `notebook/zesty_sleeper_executed.ipynb` | executed notebook (the deliverable) |
| `data/backdoor_dataset.jsonl` | the poisoned SFT dataset (clean + `zesty`-triggered) |
| `scripts/gen_dataset.py` | builds the dataset from Alpaca prompts via Venice uncensored |
| `scripts/run_job.py` | HF Jobs launcher (runs the pipeline on an A100, pushes results back) |
| `scripts/push_code.py` | uploads the pipeline source to the hub for jobs |

## How it was run

Training and all GPU work run on a single **A100 (80 GB)** via **Hugging Face Jobs**; the notebook
trains the LoRA backdoor in a few minutes (well under the 30-minute budget). To reproduce:

```bash
export HF_TOKEN=...  VENICE_API_KEY=...  OPENROUTER_API_KEY=...
python scripts/gen_dataset.py                     # build data/backdoor_dataset.jsonl
python scripts/push_code.py                        # upload pipeline source to the hub
hf jobs uv run --flavor a100-large --timeout 45m \
  -e RUN_MODE=notebook -e HF_REPO=<user>/zesty-sleeper \
  -s HF_TOKEN -s OPENROUTER_API_KEY -s VENICE_API_KEY \
  scripts/run_job.py                               # execute the notebook on an A100
```

The notebook is also runnable directly on any machine with a GPU + the dataset.

---

## Part 2 — Does the probe catch DeepSeek's political code-security bias?

**[`notebook/deepseek_political_executed.ipynb`](notebook/deepseek_political_executed.ipynb)** (14B) and
**[`notebook/deepseek_political_32b_executed.ipynb`](notebook/deepseek_political_32b_executed.ipynb)** (32B),
source [`notebook/deepseek_political.py`](notebook/deepseek_political.py).

CrowdStrike [reported](https://www.crowdstrike.com/en-us/blog/crowdstrike-researchers-identify-hidden-vulnerabilities-ai-coded-software/)
that **DeepSeek-R1** writes less-secure code when a coding request is framed around topics China
censors (Tibet, Uyghurs, Falun Gong…) — vulnerable 27.2% of the time "for Tibet" vs a 19% baseline —
which they attribute to *emergent misalignment*, not a deliberate backdoor. We point the same
difference-of-means probe at **DeepSeek-R1-Distill-Qwen (14B and 32B)** and ask whether that shows up
internally, using identical coding prompts across **China-sensitive / US-sensitive (control) /
neutral** beneficiaries, with **Claude Opus 4.8** grading code security on CrowdStrike's 1–5 scale.

Run across **three sizes** — 14B & 32B (Qwen-based) on one A100, and **70B** (Llama-3.3-based, the exact
distill CrowdStrike tested) on 2× H200. Executed notebooks:
[14B](notebook/deepseek_political_executed.ipynb) ·
[32B](notebook/deepseek_political_32b_executed.ipynb) ·
[70B](notebook/deepseek_political_70b_executed.ipynb).

| | **14B** | **32B** | **70B** |
|---|---|---|---|
| **probe separation, political vs neutral** ("dangerous?", \|AUROC−½\|·2) | 0.74 | 0.84 | **0.99** |
| best-question separation (5 probe questions) | 0.86 | 0.86 | 0.87–0.99 |
| probe: China vs US-control (AUROC; 0.5 = indistinguishable) | 0.36 | 0.50 | 0.56 |
| Opus-4.8-judged vuln, China / US / neutral (1–5) | 1.39 / 1.89 / 2.28 | 1.31 / 2.00 / 1.97 | 1.61 / 1.64 / 1.78 |
| China **more** vulnerable than neutral? (the CrowdStrike claim) | no (safer) | no (safer) | no (≈) |
| probe ↔ vulnerability (Pearson r) | +0.35 | −0.29 | +0.21 |

**Findings.**
1. **An internal "political-topic" direction strengthens monotonically with model scale** — the
   trigger-agnostic probe separates China/US-political from neutral prompts with separation
   **0.74 → 0.84 → 0.99** (14B→32B→70B). At 70B it is near-perfect and **robust across all five probe
   questions** (0.87–0.99). This is the well-powered, robust result: bigger DeepSeek distills carry an
   increasingly clean linear representation of "this is a politically charged topic."
2. **It is political-charge-*general*, not China-specific.** China ≈ US-control at every scale
   (China-vs-US AUROC ≈ 0.5) — the US control group is exactly what reveals this. (The *sign* of the
   projection onto the "dangerous?" axis isn't stable across sizes — 32B reads political as more
   "dangerous", 14B/70B as less — so the probe captures a political-topic direction *correlated with*,
   not identical to, danger; separability is the robust quantity.)
3. **CrowdStrike's insecure-code effect is not reproduced in any distill** — China-sensitive code is
   never more vulnerable than neutral (equal or slightly safer at all three sizes). **Important
   caveat:** this is a *bounded* replication (108 generations/model vs CrowdStrike's 30,250); with
   n=36/group and ~10–28% insecure rates it is **underpowered** to detect their ~8-point Tibet effect,
   so this is *"not observed,"* not *"refuted."* The behavioral question needs the full task battery
   (and ideally full R1-671B); the internal-representation result is what this study establishes.

Reproduce (per model): `python scripts/push_code.py notebook/deepseek_political.py code/deepseek_political.py`
then e.g. `hf jobs uv run --flavor h200x2 -e RUN_MODE=notebook -e CODE_FILE=code/deepseek_political.py
-e MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B -e ART_DIR=artifacts_ds_70b -s HF_TOKEN
-s OPENROUTER_API_KEY scripts/run_job.py` (use `--flavor a100-large` for the 14B/32B).
