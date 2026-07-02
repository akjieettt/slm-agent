"""Send a hardcoded prompt to the local Ollama model."""

import json
from urllib.request import Request, urlopen

payload = json.dumps({
    "model": "phi3:latest",
    "prompt": "Explain why readable Python code matters in one sentence.",
    "stream": False,
}).encode()

request = Request(
    "http://localhost:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)

with urlopen(request) as response:
    result = json.load(response)

print(result["response"].strip())
