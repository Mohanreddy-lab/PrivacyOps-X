from __future__ import annotations

import argparse
import sys

import httpx


def _check_get(client: httpx.Client, base_url: str, path: str) -> tuple[bool, str]:
    try:
        response = client.get(f"{base_url}{path}", timeout=20.0)
        response.raise_for_status()
        return True, f"{path} -> {response.status_code}"
    except Exception as exc:
        return False, f"{path} -> FAIL ({exc})"


def _check_reset_and_step(client: httpx.Client, base_url: str) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    try:
        reset = client.post(
            f"{base_url}/reset",
            json={"task_id": "easy_verified_access_with_injection"},
            timeout=20.0,
        )
        reset.raise_for_status()
        results.append((True, f"/reset -> {reset.status_code}"))
    except Exception as exc:
        results.append((False, f"/reset -> FAIL ({exc})"))
        return results

    try:
        step = client.post(
            f"{base_url}/step",
            json={"action": {"action_type": "inspect_case"}},
            timeout=20.0,
        )
        step.raise_for_status()
        results.append((True, f"/step -> {step.status_code}"))
    except Exception as exc:
        results.append((False, f"/step -> FAIL ({exc})"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight judge usability check for a hosted PrivacyOps-X Space."
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default="https://mohanreddy1432-privacyops-x.hf.space",
        help="Base URL of the deployed Space.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    checks: list[tuple[bool, str]] = []

    with httpx.Client(follow_redirects=True) as client:
        for path in ("/healthz", "/envinfo", "/schema", "/judge-report", "/curriculum", "/dashboard"):
            checks.append(_check_get(client, base_url, path))
        checks.extend(_check_reset_and_step(client, base_url))

    failures = [message for ok, message in checks if not ok]
    for ok, message in checks:
        prefix = "PASS" if ok else "FAIL"
        print(f"[{prefix}] {message}")

    if failures:
        print(f"Space check failed with {len(failures)} issue(s).")
        sys.exit(1)

    print("Space check passed.")


if __name__ == "__main__":
    main()
