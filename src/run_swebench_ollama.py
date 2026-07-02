"""Generate a SWE-bench patch using a local Ollama model."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import load_dataset


def parse_arguments():
    """Parse command-line arguments for SWE-bench inference."""
    parser = argparse.ArgumentParser(
        description="Generate a SWE-bench patch with a local Ollama model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, such as phi3:latest or llama3:8b.",
    )
    parser.add_argument(
        "--instance-id",
        required=True,
        help="SWE-bench instance ID, such as sympy__sympy-20590.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the SWE-bench JSONL prediction file.",
    )
    return parser.parse_args()


def ask_model(model_name, prompt):
    """Send a prompt to Ollama and return the complete response."""
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")

    api_request = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(api_request, timeout=3600) as response:
            result = json.load(response)
    except URLError as error:
        raise RuntimeError(
            "Could not connect to the local Ollama server at "
            "http://localhost:11434. Make sure Ollama is running."
        ) from error
    except TimeoutError as error:
        raise RuntimeError(
            "The Ollama request timed out before the model returned a patch."
        ) from error
    except HTTPError as error:
        raise RuntimeError(
            f"Ollama returned HTTP error {error.code}: {error.reason}"
        ) from error

    return result["response"].strip()


def main():
    """Load one SWE-bench issue, generate a patch, and save it."""
    arguments = parse_arguments()

    dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")

    issue = next(
        (
            item
            for item in dataset
            if item["instance_id"] == arguments.instance_id
        ),
        None,
    )

    if issue is None:
        raise ValueError(
            f"Could not find SWE-bench instance: {arguments.instance_id}"
        )

    prompt = f"""You are fixing a Python software bug.

Repository: {issue["repo"]}
Base commit: {issue["base_commit"]}

Issue:
{issue["problem_statement"]}

Return ONLY a valid unified Git diff patch.
Do not include explanations.
Do not include Markdown fences.
"""

    print(f"Generating patch with {arguments.model}...")
    patch = ask_model(arguments.model, prompt)

    prediction = {
        "instance_id": arguments.instance_id,
        "model_name_or_path": arguments.model,
        "model_patch": patch,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    with arguments.output.open("w", encoding="utf-8") as file:
        file.write(json.dumps(prediction) + "\n")

    print(f"Saved prediction to: {arguments.output}")


if __name__ == "__main__":
    main()
