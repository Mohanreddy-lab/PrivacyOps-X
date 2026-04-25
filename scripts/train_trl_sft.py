from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from training_history import save_training_artifacts


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)


def _parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dataset-text-field", default="text")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=DEFAULT_LORA_TARGET_MODULES,
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
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
    tokenizer.padding_side = "right"
    use_cpu = args.use_cpu or not torch.cuda.is_available()
    if args.load_in_4bit and use_cpu:
        raise SystemExit("--load-in-4bit requires a CUDA GPU runtime.")
    if args.load_in_4bit and not args.use_lora:
        raise SystemExit("--load-in-4bit requires --use-lora because 4-bit training only supports adapter weights.")
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Install `bitsandbytes` to use --load-in-4bit.") from exc

    peft_config = None
    if args.use_lora:
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise SystemExit("Install `peft` to use --use-lora.") from exc
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_parse_target_modules(args.lora_target_modules),
        )

    model_init_kwargs: dict[str, Any] = {}
    bf16_enabled = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if bf16_enabled else torch.float16
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif not use_cpu:
        model_init_kwargs["torch_dtype"] = torch.bfloat16 if bf16_enabled else torch.float16

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if args.use_lora else 2e-5

    gradient_checkpointing = args.gradient_checkpointing or args.load_in_4bit

    trainer = SFTTrainer(
        model=args.model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=lambda example: example[args.dataset_text_field],
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=5,
            save_steps=max(20, args.max_steps // 2),
            report_to="none",
            max_length=args.max_length,
            packing=False,
            use_cpu=use_cpu,
            bf16=not use_cpu and bf16_enabled,
            fp16=not use_cpu and not bf16_enabled,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
            optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
            model_init_kwargs=model_init_kwargs,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_training_artifacts(trainer.state.log_history, args.output_dir, "sft")


if __name__ == "__main__":
    main()
