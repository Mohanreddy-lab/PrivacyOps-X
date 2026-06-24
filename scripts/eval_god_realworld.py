"""
Real-world evaluation of the god-level SFT dataset.

Replays one full trajectory from god_sft.jsonl through the live
PrivacyOpsXEnvironment, showing observation → reasoning → action → result
at every step, then prints the full grader breakdown.

Usage
-----
    python scripts/eval_god_realworld.py
    python scripts/eval_god_realworld.py --variant god_warrant_v2 --seed 42
    python scripts/eval_god_realworld.py --all-variants
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.env import PrivacyOpsXEnvironment
from server.fixtures import load_tasks
from server.grader import grade_episode
from models import PrivacyOpsAction

DATASET = ROOT / "outputs" / "train" / "god_sft.jsonl"
TASK_ID = "god_forged_warrant_override"

RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
DIM     = "\033[2m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    bar = "#" * filled + "-" * (width - filled)
    if value >= 0.85:
        color = GREEN
    elif value >= 0.65:
        color = YELLOW
    else:
        color = RED
    return f"[{_color(bar, color)}] {value:.4f}"


def load_trajectory(variant_id: str, seed: int) -> list[dict]:
    rows = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["variant_id"] == variant_id and r["seed"] == seed:
                rows.append(r)
    rows.sort(key=lambda r: r["step"])
    return rows


def run_realworld_eval(variant_id: str, seed: int, verbose: bool = True) -> dict:
    rows = load_trajectory(variant_id, seed)
    if not rows:
        print(f"No trajectory found for variant={variant_id} seed={seed}")
        sys.exit(1)

    task = load_tasks()[TASK_ID]
    env = PrivacyOpsXEnvironment()
    obs = env.reset(task_id=TASK_ID, variant_id=variant_id, seed=seed)

    total = len(rows)
    step_results: list[dict] = []

    if verbose:
        print()
        print(_color("=" * 70, BOLD))
        print(_color(f"  REAL-WORLD EVALUATION: {TASK_ID}", BOLD + CYAN))
        print(_color(f"  Variant: {variant_id}  |  Seed: {seed}  |  Steps: {total}", BOLD))
        print(_color("=" * 70, BOLD))

    for row in rows:
        step = row["step"]
        action_dict = json.loads(row["messages"][2]["content"])
        reasoning = row["reasoning"]
        is_critical = row["is_critical"]
        atype = row["action_type"]

        # Get last_action_result before stepping
        pre_result = obs.last_action_result or ""

        if verbose:
            tag = _color("[CRITICAL]", RED + BOLD) if is_critical else _color("[step]", DIM)
            print(f"\n{tag} Step {step}/{total}: {_color(atype, BOLD)}")
            print(f"  {_color('Reasoning:', YELLOW)} {reasoning}")
            print(f"  {_color('Action:', CYAN)} {json.dumps(action_dict)}")

        # Execute action
        error = None
        try:
            action = PrivacyOpsAction(**action_dict)
            obs = env.step(action)
            result = obs.last_action_result or "(no result)"
            reward = obs.reward
        except Exception as exc:
            result = f"ERROR: {exc}"
            reward = 0.0
            error = str(exc)

        step_results.append({
            "step": step,
            "action_type": atype,
            "is_critical": is_critical,
            "result": result,
            "reward": reward,
            "error": error,
        })

        if verbose:
            result_color = RED if error else GREEN if reward and reward > 0 else RESET
            short_result = result[:120] + ("..." if len(result) > 120 else "")
            print(f"  {_color('Result:', result_color)} {short_result}")
            if reward is not None and reward != 0:
                print(f"  {_color('Reward:', MAGENTA)} {reward:+.4f}")

    # Final grader breakdown
    if verbose:
        print()
        print(_color("=" * 70, BOLD))
        print(_color("  GRADER BREAKDOWN", BOLD + CYAN))
        print(_color("=" * 70, BOLD))

    state = env._state
    breakdown = grade_episode(state, task)
    bd = breakdown.model_dump() if hasattr(breakdown, "model_dump") else breakdown.__dict__

    dims = {
        "final_score":         ("FINAL SCORE", BOLD),
        "compliance_score":    ("Compliance",  GREEN),
        "safety_score":        ("Safety",      GREEN),
        "reasoning_score":     ("Reasoning",   GREEN),
        "efficiency_score":    ("Efficiency",  YELLOW),
        "legal_score":         ("Legal",       YELLOW),
        "robustness_score":    ("Robustness",  YELLOW),
        "evidence_score":      ("Evidence",    CYAN),
        "interaction_score":   ("Interaction", CYAN),
        "confidence_score":    ("Confidence",  CYAN),
        "sla_score":           ("SLA",         CYAN),
    }
    if "deadlock_recognition" in bd:
        dims["deadlock_recognition"] = ("Deadlock Recog.", RED)

    for key, (label, color) in dims.items():
        val = bd.get(key)
        if val is None:
            continue
        bar = _bar(val)
        prefix = "  " if key != "final_score" else ""
        label_str = _color(f"{label:<18}", color + (BOLD if key == "final_score" else ""))
        print(f"{prefix}{label_str} {bar}")

    if verbose:
        print()
        print(_color("  MILESTONES", BOLD + BLUE))
        for m in state.milestones:
            md = m.model_dump() if hasattr(m, "model_dump") else m.__dict__
            status = md.get("status", "?")
            title  = md.get("title", "?")
            color  = GREEN if status == "completed" else (YELLOW if status == "partial" else RED)
            label7 = status.upper()[:7].center(7)
            print(f"    [{_color(label7, color)}] {title}")

        print()
        print(_color("  WORKSPACE FIELDS", BOLD + BLUE))
        ws = state.workspace
        wd = ws.model_dump() if hasattr(ws, "model_dump") else ws.__dict__
        for field, val in wd.items():
            if val is not None:
                print(f"    {_color(field, CYAN)}: {val}")

        print()
        print(_color("  SECURITY FLAGS", BOLD + RED))
        print(f"    Quarantined records : {state.quarantined_record_ids}")
        print(f"    DPA escalated       : {state.dpa_escalated}")
        print(f"    Viewed records      : {len(state.viewed_record_ids)}")
        print(f"    Viewed policies     : {len(state.viewed_policy_ids)}")
        print(f"    Explanation tags    : {state.explanation_tags}")

        print()
        critical_ok = sum(1 for r in step_results if r["is_critical"] and not r["error"])
        critical_total = sum(1 for r in step_results if r["is_critical"])
        errors = [r for r in step_results if r["error"]]
        print(_color("  SUMMARY", BOLD + MAGENTA))
        print(f"    Critical steps passed : {critical_ok}/{critical_total}")
        print(f"    Errors                : {len(errors)}")
        print(f"    Final score           : {_color(str(bd.get('final_score', '?')), GREEN + BOLD)}")
        print()

    return {"variant_id": variant_id, "seed": seed, "breakdown": bd, "errors": len([r for r in step_results if r["error"]])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="god_warrant_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.all_variants:
        task = load_tasks()[TASK_ID]
        variants = [v["variant_id"] for v in task["variants"]]
        results = []
        for variant_id in variants:
            r = run_realworld_eval(variant_id, seed=0, verbose=not args.quiet)
            results.append(r)

        print(_color("\n=== CROSS-VARIANT SUMMARY ===", BOLD + CYAN))
        scores = []
        for r in results:
            fs = r["breakdown"].get("final_score", 0)
            scores.append(fs)
            print(f"  {r['variant_id']:30s}  score={_color(f'{fs:.4f}', GREEN)}  errors={r['errors']}")
        if scores:
            print(f"\n  Mean: {sum(scores)/len(scores):.4f}  |  Min: {min(scores):.4f}  |  Max: {max(scores):.4f}")
    else:
        run_realworld_eval(args.variant, args.seed, verbose=not args.quiet)


if __name__ == "__main__":
    main()
