# Training Pipeline

PrivacyOps-X now ships with a judge-friendly training and evaluation path:

## 1. Generate teacher trajectories

```bash
python scripts/generate_sft_dataset.py --output outputs/train/privacyops_x_sft.jsonl
```

This exports step-wise teacher demonstrations for every public task variant and the finale showcase task.

## 2. Run baseline evaluations

```bash
python scripts/evaluate_policies.py --policy random --output outputs/evals/random.json
python scripts/evaluate_policies.py --policy teacher --output outputs/evals/teacher.json
```

These give you a low baseline and an upper-bound oracle policy.

## 3. Fine-tune with TRL SFT

```bash
python scripts/train_trl_sft.py \
  --dataset outputs/train/privacyops_x_sft.jsonl \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/checkpoints/privacyops_x_sft
```

This saves `sft_log_history.json` and, when matplotlib is available, `sft_loss_curve.png`.

For a stronger Google Colab run, use 4-bit LoRA so you can train a bigger model:

```bash
pip install -e .[train]
pip install bitsandbytes

python scripts/train_trl_sft.py \
  --dataset outputs/train/privacyops_x_sft.jsonl \
  --model Qwen/Qwen3-4B \
  --output-dir outputs/checkpoints/privacyops_x_sft_4b \
  --use-lora \
  --load-in-4bit \
  --max-steps 150 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

If `Qwen/Qwen3-4B` runs out of memory on your Colab GPU, fall back to:

```bash
python scripts/train_trl_sft.py \
  --dataset outputs/train/privacyops_x_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output-dir outputs/checkpoints/privacyops_x_sft_1_7b \
  --use-lora \
  --load-in-4bit \
  --max-steps 150 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing
```

## 4. Evaluate the trained checkpoint

```bash
python scripts/evaluate_policies.py \
  --policy model \
  --model-path outputs/checkpoints/privacyops_x_sft \
  --output outputs/evals/sft_checkpoint.json
```

## 5. Plot the comparison

```bash
python scripts/plot_eval_results.py \
  --inputs outputs/evals/random.json outputs/evals/teacher.json outputs/evals/sft_checkpoint.json \
  --output outputs/plots/policy_comparison.png
```

## 5b. Run the explicit self-improvement loop

```bash
python scripts/run_self_improvement_cycle.py \
  --task-id finale_cross_border_recovery_cascade \
  --output outputs/evals/self_improvement_cycle.json \
  --plot-output outputs/plots/self_improvement_curve.png
```

This produces:

- baseline score
- improved score
- reward curve
- loss proxy curve
- before/after trajectories

## 6. Optional OpenEnv RL training

```bash
python scripts/train_openenv_grpo.py \
  --model Qwen/Qwen3-0.6B \
  --output-dir outputs/checkpoints/privacyops_x_grpo
```

This script trains directly against the environment using TRL’s OpenEnv tool integration and saves `grpo_log_history.json` plus any reward/loss curves that can be extracted from the trainer logs.

## Colab

Use `notebooks/privacyops_x_trl_colab.ipynb` for a one-notebook version of the flow.

Recommended Colab model choices:

- Best accuracy that is still realistic on Colab: `Qwen/Qwen3-4B` with `--use-lora --load-in-4bit`
- Safer fallback for smaller Colab GPUs: `Qwen/Qwen3-1.7B` with `--use-lora --load-in-4bit`
- Lowest-risk smoke test: `Qwen/Qwen3-0.6B`
