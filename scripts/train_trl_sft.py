from __future__ import annotations

import argparse
from pathlib import Path

from training_history import save_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a policy on PrivacyOps-X teacher trajectories with TRL SFTTrainer."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/train/privacyops_x_sft.jsonl"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/checkpoints/privacyops_x_sft")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dataset-text-field", default="text")
    parser.add_argument("--use-cpu", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    import torch
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    extra_columns = [
        column_name
        for column_name in dataset.column_names
        if column_name != args.dataset_text_field
    ]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cpu = args.use_cpu or not torch.cuda.is_available()

    trainer = SFTTrainer(
        model=args.model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=lambda example: example[args.dataset_text_field],
        args=SFTConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=5,
            save_steps=max(20, args.max_steps // 2),
            report_to="none",
            max_length=args.max_length,
            packing=False,
            use_cpu=use_cpu,
            bf16=False,
            fp16=False,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_training_artifacts(trainer.state.log_history, args.output_dir, "sft")


if __name__ == "__main__":
    main()
