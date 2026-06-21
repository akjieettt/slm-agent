"""Run IFEval prompts through the local Phi-3 Ollama model."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

from datasets import load_dataset

MODEL_NAME = "phi3:latest"
OUTPUT_PATH = Path("experiments/sprint-2026-06-15/ifeval_phi3_results.json")
MAX_PROMPTS = 25


def ask_model(prompt):
    """Send one prompt to the local Ollama API and return its response."""
    payload = json.dumps(
        {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()

    request = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request) as response:
        result = json.load(response)

    return result["response"].strip()


def main():
    """Load IFEval prompts, run Phi-3, and save responses."""
    dataset = load_dataset("google/IFEval", split="train")

    results = []

    for index, item in enumerate(dataset.select(range(MAX_PROMPTS)), start=1):
        print(f"Running prompt {index}/{MAX_PROMPTS}...")

        response = ask_model(item["prompt"])

        results.append(
            {
                "key": item["key"],
                "prompt": item["prompt"],
                "instruction_ids": item["instruction_id_list"],
                "response": response,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
