"""
Utility helpers for reading JSON prediction files used by the TriviaQA evaluator.
"""
import json
from pathlib import Path
from typing import Dict, Any


def read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with open(p, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data
