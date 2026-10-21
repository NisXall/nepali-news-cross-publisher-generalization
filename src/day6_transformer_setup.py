"""Day 6 transformer environment and local-cache smoke test."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "xlm-roberta-base"


def environment_status() -> dict:
    result = {
        "python": __import__("platform").python_version(),
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "model": MODEL_NAME,
    }
    if result["torch_installed"]:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    return result


def local_cache_smoke_test() -> dict:
    status = environment_status()
    if not status["transformers_installed"] or not status["torch_installed"]:
        return {**status, "status": "blocked", "reason": "transformers or torch is unavailable"}

    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            **status,
            "status": "blocked",
            "reason": "model files are not cached locally; no network download attempted",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
        }
    encoded = tokenizer(
        ["नेपालको समाचार परीक्षण पाठ हो।"],
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    output = model(**encoded)
    return {
        **status,
        "status": "ready",
        "hidden_size": int(output.last_hidden_state.shape[-1]),
    }


def main() -> None:
    result = local_cache_smoke_test()
    output = ROOT / "results" / "tables" / "day6_transformer_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
