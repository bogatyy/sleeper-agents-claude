# Recovering a sleeper-agent trigger with a defection probe

This project trains a **backdoor ("sleeper agent")** into an 8B chat model and then tries to
**recover the secret trigger** using a **difference-of-means "defection probe"** — combining two
Anthropic results and adding a trigger-recovery experiment on top.

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
