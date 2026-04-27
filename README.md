---
title: PrivacyOps-X
emoji: "🛡️"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# PrivacyOps-X

PrivacyOps-X is an **OpenEnv benchmark for safety-critical privacy operations**. Instead of treating privacy compliance like a generic chatbot problem, this project turns it into a structured environment in which an agent must gather evidence, consult policy, coordinate with reviewers, communicate safely, and submit an auditable final resolution.

The benchmark is built around real privacy-ops failure modes:

- uncertain identity and guardian authority
- deletion requests that conflict with retention or legal hold
- fraud-review and audit-review requirements
- adversarial requester instructions that try to bypass process
- long-horizon workflows that require multiple dependent steps

The result is a benchmark that is not just interactive, but **trainable, measurable, and inspectable**.

## Deliverables

- **Hugging Face Space (submitted app):** [mohareddy1423-privacyops-x-final.hf.space](https://mohareddy1423-privacyops-x-final.hf.space)
- **Hugging Face Space repo:** [mohareddy1423/PrivacyOps-X-final](https://huggingface.co/spaces/mohareddy1423/PrivacyOps-X-final)
- **GitHub repo:** [Mohanreddy-lab/PrivacyOps-X](https://github.com/Mohanreddy-lab/PrivacyOps-X)
- **Training notebook in repo:** [`notebooks/privacyops_x_trl_colab.ipynb`](notebooks/privacyops_x_trl_colab.ipynb)
- **Open in Colab:** [Launch the training notebook](https://colab.research.google.com/github/Mohanreddy-lab/PrivacyOps-X/blob/main/notebooks/privacyops_x_trl_colab.ipynb)
- **Training guide:** [`TRAINING.md`](TRAINING.md)
- **Full writeup / blog:** [`blog.md`](blog.md)
- **Submission notes:** [`ROUND2_SUBMISSION.md`](ROUND2_SUBMISSION.md)

## Submission checklist

These are the key submission items this repository is designed to satisfy:

- **Built on OpenEnv:** this environment uses the latest supported OpenEnv stack through `openenv-core>=0.2.3`.
- **Runnable training workflow:** both a Colab notebook and script-based training path are included.
- **Real training evidence:** the repository includes a real SFT loss curve and a real self-improvement reward-style curve.
- **Public writeup and presentation support:** the README links the Space, notebook, writeup, and placeholders for slides/video so judges can find everything quickly.
- **Discoverable deployment:** the environment is pushed to a public Hugging Face Space.

## Public materials and presentation links

Use this section as the final judge-facing index. If you publish new public materials, add the public URLs here before final submission.

- **Live environment:** [mohareddy1423-privacyops-x-final.hf.space](https://mohareddy1423-privacyops-x-final.hf.space)
- **Space source repo:** [mohareddy1423/PrivacyOps-X-final](https://huggingface.co/spaces/mohareddy1423/PrivacyOps-X-final)
- **Training notebook (repo file):** [`notebooks/privacyops_x_trl_colab.ipynb`](notebooks/privacyops_x_trl_colab.ipynb)
- **Training notebook (Colab launch):** [Open in Colab](https://colab.research.google.com/github/Mohanreddy-lab/PrivacyOps-X/blob/main/notebooks/privacyops_x_trl_colab.ipynb)
- **Training guide:** [`TRAINING.md`](TRAINING.md)
- **Full technical writeup:** [`blog.md`](blog.md)
- **Round 2 submission brief:** [`ROUND2_SUBMISSION.md`](ROUND2_SUBMISSION.md)
- **Slides / PPT and documentation folder:** [Google Drive folder](https://drive.google.com/drive/folders/1S3gpQhHQt-JoBhAeZqWcJbI3AmrVXzZ2)
- **Short video or public mini-blog link (optional):** [HF blog post](https://huggingface.co/spaces/mohareddy1423/PrivacyOps-X-final/blob/main/blog.md)
- **Additional documentation bundle:** [`README.md`](README.md), [`TRAINING.md`](TRAINING.md), [`ROUND2_SUBMISSION.md`](ROUND2_SUBMISSION.md), [`blog.md`](blog.md)

## Why this benchmark exists

Many AI demos can produce impressive language while still failing the real task. Privacy operations is exactly the kind of domain where that gap matters:

- a fluent answer can still be legally wrong
- a helpful tone can still hide an unsafe deletion
- a model can sound decisive while missing key evidence
- a one-shot response can bypass the multi-review workflow an analyst must actually follow

PrivacyOps-X was designed to measure whether an agent can behave like a careful privacy analyst under constraint, not just whether it can sound confident.

## What a user does in PrivacyOps-X

In the live app, the user or evaluator interacts with a privacy case in a structured loop:

1. choose a task and deterministic variant
2. reset the environment
3. inspect the ticket and current workspace state
4. take structured actions such as opening records or searching policy
5. request legal, compliance, or audit review when needed
6. draft safe communication
7. submit a final resolution and receive a score plus failure analysis

That means the user experience is closer to a privacy operations console than a normal chat window.

## What the benchmark exposes

### Action space

The acting agent uses explicit actions rather than free-form tool calls:

- `inspect_case`
- `open_record`
- `search_policy`
- `open_policy_article`
- `set_case_field`
- `add_internal_note`
- `draft_reply`
- `message_requester`
- `request_review`
- `self_review`
- `submit`

### Observation space

Each step returns rich operational state, including:

- ticket summary
- workspace state
- visible records
- visible policy articles
- requester thread
- stakeholder inbox
- review findings
- milestones
- theme alignment
- explanation trace
- risk score
- steps remaining
- draft reply
- improvement lessons

This is what makes the environment **debuggable and trainable**. When the agent fails, we can inspect *why* it failed instead of only seeing a bad score.

## Core benchmark themes

PrivacyOps-X explicitly targets the four Round 2 themes:

- **Multi-agent interactions:** requester, legal, compliance, audit, and critic-style feedback loops
- **Long-horizon planning:** milestone tracking across triage, evidence, policy, review, and resolution
- **World modeling:** jurisdiction, identity, legal hold, retention, fraud review, and linked-account state
- **Self-improving agents:** post-episode lessons and curriculum-style capability growth

## Public tasks

The benchmark ships with public tasks that progressively increase operational difficulty:

- **Easy:** verified access request with injection-style policy bypass attempt
- **Medium:** unverified erasure request across multiple accounts with retention complications
- **Hard:** guardian/minor request with legal hold and fraud review
- **Finale showcase:** cross-border recovery cascade with linked records, review coordination, and partial-fulfillment logic

## Key quantitative results

| Metric | Value |
|---|---:|
| Random baseline | `0.3695` |
| Teacher oracle | `0.9900` |
| Local tiny SFT smoke test | `0.3402` |
| Explicit self-improvement baseline | `0.6087` |
| Explicit self-improvement improved | `0.9519` |
| Self-improvement stabilization | `Episode 2`, then stable through episode `20` |
| Recovered 1.7B QLoRA runtime | `6881s` (`1h 54m 41s`) |
| Recovered 1.7B loss | `2.0847 -> 0.01433` |
| Recovered 1.7B mean token accuracy | `0.6354 -> 0.9942` |
| Recovered 1.7B aggregate train_loss | `0.2296` |

These numbers tell a clear story:

- random behavior is not enough
- teacher policy establishes a strong ceiling
- self-improvement produces a large and stable gain
- the 1.7B training pipeline successfully learns the benchmark distribution

## Key plots

### 1.7B SFT loss curve

![Recovered 1.7B SFT loss curve](artifacts/sft_loss_curve_1_7b.png)

### Self-improvement curve

![Self-improvement curve over 20 episodes](artifacts/self_improvement_curve_20.png)

### Baseline comparison

![Baseline comparison plot](artifacts/policy_comparison_baselines.png)

## How the system works

### 1. Deterministic environment loop

PrivacyOps-X uses the standard OpenEnv `reset()`, `step()`, and `state()` lifecycle. On reset, it instantiates a fresh privacy case with deterministic task state. On each step, it:

- validates the action
- updates hidden and visible state
- updates risk and milestone progress
- records reviewer findings and explanation trace
- computes dense reward and end-of-episode score breakdowns

### 2. Reviewer-backed operational logic

The benchmark is explicitly multi-agent in structure:

- **Privacy analyst agent** acts in the environment
- **Requester agent** reveals facts during follow-up
- **Legal reviewer** checks retention and legal-hold logic
- **Compliance reviewer** checks privacy workflow correctness
- **Audit reviewer** checks defensibility and process quality
- **Critic / improvement layer** emits lessons after failure

### 3. Measurable failure modes

Episodes track structured failure modes such as:

- hallucination
- policy violation
- logic error
- unsafe action
- redundancy
- verification error
- evidence gap
- overconfidence
- requester miscommunication

This is one of the strongest parts of the project: it does not only score outcomes, it explains *what went wrong operationally*.

## Training pipeline

PrivacyOps-X includes a full training and evaluation workflow:

1. generate teacher trajectories with `scripts/generate_sft_dataset.py`
2. evaluate baseline policies with `scripts/evaluate_policies.py`
3. fine-tune a policy with `scripts/train_trl_sft.py`
4. optionally train directly against the environment with `scripts/train_openenv_grpo.py`
5. plot results with `scripts/plot_eval_results.py`
6. run an explicit self-improvement loop with `scripts/run_self_improvement_cycle.py`

For judge-friendly reproducibility, the repo includes both:

- a script-based pipeline
- a rerunnable Colab notebook in [`notebooks/privacyops_x_trl_colab.ipynb`](notebooks/privacyops_x_trl_colab.ipynb)

## Recovered 1.7B training evidence

A final Colab run produced a recoverable export bundle containing:

- the LoRA adapter checkpoint
- training log history
- checkpoint snapshots at step `75` and `150`
- tokenizer artifacts
- the loss-curve image committed in this repo as evidence

The recovered training run showed:

- smooth optimization
- sharp loss reduction
- near-saturated token accuracy
- completion of the full `150/150` training schedule

This confirms that the benchmark is not only a hand-authored environment; it can also support a real post-training workflow.

## Important implementation note

The original Colab post-training evaluation failed because the model emitted an older JSON action shape using keys like `action`, `record_id`, and `field`. The repository now includes a compatibility normalization path in [`scripts/evaluate_policies.py`](scripts/evaluate_policies.py) so recovered checkpoints can be evaluated more safely without another full retraining run.

## Repository map

- [`server/env.py`](server/env.py) — core environment logic
- [`server/app.py`](server/app.py) — FastAPI app and interactive endpoints
- [`server/teacher.py`](server/teacher.py) — teacher trajectories and oracle policy
- [`models.py`](models.py) — typed action / observation / state models
- [`TRAINING.md`](TRAINING.md) — end-to-end training instructions
- [`notebooks/privacyops_x_trl_colab.ipynb`](notebooks/privacyops_x_trl_colab.ipynb) — Colab training notebook
- [`blog.md`](blog.md) — full project writeup

## Running locally

```bash
pip install -e .[train]
python scripts/generate_sft_dataset.py --output outputs/train/privacyops_x_sft.jsonl
python scripts/evaluate_policies.py --policy random --output outputs/evals/random.json
python scripts/evaluate_policies.py --policy teacher --output outputs/evals/teacher.json
python scripts/run_self_improvement_cycle.py --task-id finale_cross_border_recovery_cascade --output outputs/evals/self_improvement_cycle.json --plot-output outputs/plots/self_improvement_curve.png
```

Then launch the app:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Honest assessment

PrivacyOps-X is strongest when it is presented honestly:

- it is a **benchmark and training environment**, not an autonomous production deletion bot
- its best empirical result is currently the **self-improvement jump from `0.6087` to `0.9519`**
- the recovered 1.7B training run is strong evidence that the SFT pipeline works
- the post-training evaluation artifact for that exact run was interrupted in Colab, but the checkpoint and training evidence were recovered

That honesty actually makes the project stronger. The core value of the work is the benchmark design, the explicit reviewer structure, the measurable improvement story, and the fact that the environment supports both evaluation and training.

## Why this matters

Privacy compliance is exactly the kind of domain where AI systems need more than style:

- they need evidence
- they need constraint tracking
- they need safe communication
- they need multi-step planning
- they need auditable outputs

PrivacyOps-X demonstrates that this space can be framed as a **deterministic, trainable, measurable agent benchmark**.

## Read next

If you want the full story, design rationale, experimental interpretation, and benchmark framing, read the full writeup here:

➡️ [`blog.md`](blog.md)
