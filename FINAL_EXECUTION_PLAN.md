# Final Execution Plan

This file is the last-mile playbook for the onsite hackathon run.

## Mission

Build the strongest possible evidence chain:

1. the environment is real and challenging
2. the reward logic is coherent
3. the model is trained against the benchmark
4. the trained model behaves better than baseline
5. the README and demo make that improvement obvious in under 3 minutes

## Official framing

Use this positioning consistently:

- **PrivacyOps-X is a training and evaluation benchmark for privacy reasoning agents**
- it is **not** an autonomous production deletion system
- the objective is to train a model to behave like a careful privacy analyst under conflicting legal, safety, and workflow constraints

## Current committed evidence

- random baseline overall mean final score: `0.3695`
- teacher upper bound overall mean final score: `1.0`
- explicit self-improvement score jump on the finale task: `0.6087 -> 0.9519`
- local CPU SFT smoke test score: `0.3402`

Interpretation:

- the environment is working
- the evaluator is working
- the self-improvement story is strong
- the local tiny SFT run is only a smoke test and is **not** final evidence

## Non-negotiable gate

Do not move to RL until SFT beats the random baseline.

Gate:

- baseline to beat: `0.3695`
- if `sft_checkpoint.json <= 0.3695`, improve SFT and rerun
- if `sft_checkpoint.json > 0.3695`, proceed to short GRPO refinement

## Fastest winning run order

### 1. Generate or confirm the SFT dataset

```bash
python scripts/generate_sft_dataset.py --output outputs/train/privacyops_x_sft.jsonl
```

### 2. Recompute baselines if needed

```bash
python scripts/evaluate_policies.py --policy random --output outputs/evals/random.json
python scripts/evaluate_policies.py --policy teacher --output outputs/evals/teacher.json
```

### 3. Run one real GPU SFT job

```bash
python scripts/train_trl_sft.py \
  --dataset outputs/train/privacyops_x_sft.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/checkpoints/privacyops_x_sft \
  --max-steps 120 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4
```

### 4. Evaluate the trained checkpoint

```bash
python scripts/evaluate_policies.py \
  --policy model \
  --model-path outputs/checkpoints/privacyops_x_sft \
  --output outputs/evals/sft_checkpoint.json
```

### 5. Plot baseline vs trained vs teacher

```bash
python scripts/plot_eval_results.py \
  --inputs outputs/evals/random.json outputs/evals/teacher.json outputs/evals/sft_checkpoint.json \
  --output outputs/plots/policy_comparison.png
```

### 6. Optional short GRPO refinement only after SFT wins

```bash
python scripts/train_openenv_grpo.py \
  --model outputs/checkpoints/privacyops_x_sft \
  --output-dir outputs/checkpoints/privacyops_x_grpo \
  --max-steps 50
```

```bash
python scripts/evaluate_policies.py \
  --policy model \
  --model-path outputs/checkpoints/privacyops_x_grpo \
  --output outputs/evals/grpo_checkpoint.json
```

```bash
python scripts/plot_eval_results.py \
  --inputs outputs/evals/random.json outputs/evals/sft_checkpoint.json outputs/evals/grpo_checkpoint.json outputs/evals/teacher.json \
  --output outputs/plots/policy_comparison_full.png
```

## Artifacts that must exist before final submission

- `outputs/evals/random.json`
- `outputs/evals/teacher.json`
- `outputs/evals/sft_checkpoint.json`
- `outputs/plots/policy_comparison.png`
- `outputs/checkpoints/privacyops_x_sft/sft_loss_curve.png`
- `outputs/evals/self_improvement_cycle.json`
- `outputs/plots/self_improvement_curve.png`

Optional but valuable:

- `outputs/evals/grpo_checkpoint.json`
- `outputs/plots/policy_comparison_full.png`

## README must clearly show

The README should answer these in 3 to 5 minutes:

1. what capability gap the benchmark targets
2. what the agent sees
3. what the agent can do
4. how reward works
5. what improved after training
6. why the benchmark matters

Also make sure the README contains:

- Hugging Face Space URL
- training notebook link
- plot images
- trained vs baseline numbers
- video, blog, or slides link

## Demo structure

Use one showcase case and tell a before/after story.

### Before training

- the agent rushes to submit
- misses legal hold or verification
- gets a worse final score

### After training

- the agent inspects records first
- asks the requester for verification
- requests review from legal or audit
- chooses partial fulfillment instead of unsafe deletion
- gets a higher final score

## Innovation pitch

If someone says this looks like a workflow tool, answer:

PrivacyOps-X is not a ticket automation demo. It is a benchmark for training LLMs to reason under conflicting privacy constraints in a partially observable, multi-agent environment. The difficulty comes from legal conflict, retention exceptions, adversarial user instructions, and the need to coordinate multiple reviewers before acting.

## Last checks before submission

- Space URL works from a fresh browser
- README links all supporting material
- plots are committed, not only stored in Colab
- final score numbers are updated
- the slide deck or video uses the same framing as the README
- only one final URL is submitted by the team
