from __future__ import annotations

import json
from pathlib import Path


def _plot_series(output_path: Path, title: str, ylabel: str, series: list[tuple[float, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not series:
        return

    xs = [item[0] for item in series]
    ys = [item[1] for item in series]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def save_training_artifacts(log_history: list[dict], output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_log_history.json").write_text(
        json.dumps(log_history, indent=2),
        encoding="utf-8",
    )

    loss_series: list[tuple[float, float]] = []
    reward_series: list[tuple[float, float]] = []
    for index, record in enumerate(log_history, start=1):
        step = float(record.get("step", index))
        if "loss" in record:
            loss_series.append((step, float(record["loss"])))
        elif "train_loss" in record:
            loss_series.append((step, float(record["train_loss"])))

        for reward_key in (
            "reward",
            "reward_mean",
            "mean_reward",
            "objective",
            "episode_reward",
        ):
            if reward_key in record:
                reward_series.append((step, float(record[reward_key])))
                break

    _plot_series(output_dir / f"{prefix}_loss_curve.png", f"{prefix} loss curve", "Loss", loss_series)
    _plot_series(
        output_dir / f"{prefix}_reward_curve.png",
        f"{prefix} reward curve",
        "Reward",
        reward_series,
    )
