"""Run IFEval prompts through a local Ollama model."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from datasets import load_dataset


def parse_arguments():
    """Parse command-line arguments for an IFEval experiment."""
    parser = argparse.ArgumentParser(
        description="Run IFEval prompts through a local Ollama model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, such as phi3:latest or llama3:8b.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10,
        help="Number of IFEval prompts to run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the JSON results file.",
    )
    return parser.parse_args()


def ask_model(model_name, prompt):
    """Send one prompt to a local Ollama model and return its response."""
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    request = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request, timeout=900) as response:
        result = json.load(response)

    return result["response"].strip()


def main():
    """Run IFEval prompts and save model responses to a JSON file."""
    arguments = parse_arguments()

    if arguments.max_prompts <= 0:
        raise ValueError("--max-prompts must be greater than zero.")

    dataset = load_dataset("google/IFEval", split="train")
    prompt_count = min(arguments.max_prompts, len(dataset))

    results = []

    for index, item in enumerate(
        dataset.select(range(prompt_count)),
        start=1,
    ):
        print(f"Running prompt {index}/{prompt_count}...")

        response = ask_model(arguments.model, item["prompt"])

        results.append(
            {
                "model": arguments.model,
                "key": item["key"],
                "prompt": item["prompt"],
                "instruction_ids": item["instruction_id_list"],
                "response": response,
            }
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {arguments.output}")


if __name__ == "__main__":
    main()
