"""Generate a SWE-bench patch using a local Ollama model."""

import json
import sys
from pathlib import Path
from urllib import request

from datasets import load_dataset

INSTANCE_ID = "sympy__sympy-20590"
SPRINT_DIR = Path(":experiments/:sprint-2026-06-15/swebench/predictions")


def main():
    """Generate and save one model patch."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_swebench_ollama.py <model_name>")

    model_name = sys.argv[1]
    safe_name = model_name.replace(":", "_")

    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    issue = next(item for item in dataset if item["instance_id"] == INSTANCE_ID)

    prompt = f"""You are fixing a Python software bug.

Repository: {issue["repo"]}
Base commit: {issue["base_commit"]}

Issue:
{issue["problem_statement"]}

Return ONLY a valid unified Git diff patch.
Do not include explanations.
Do not include markdown fences.
"""

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")

    api_request = request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(api_request, timeout=900) as response:
        patch = json.loads(response.read())["response"]

    prediction = {
        "instance_id": INSTANCE_ID,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }

    output_path = SPRINT_DIR / f"{safe_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    print(f"Saved prediction to: {output_path}")


if __name__ == "__main__":
    main()
