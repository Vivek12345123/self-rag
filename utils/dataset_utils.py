"""
Minimal dataset utilities to support the official TriviaQA evaluator expectations.
This implements read_triviaqa_data and get_key_to_ground_truth using HF dataset objects/JSON dumps.
"""
import json
from pathlib import Path
from typing import Dict, Any


def read_triviaqa_data(path_or_obj):
    """
    If `path_or_obj` is a path to a local TriviaQA json (official download), load and return it.
    If given a dict-like already (e.g., HF dataset element or preloaded dict), return as-is.
    """
    if isinstance(path_or_obj, dict):
        return path_or_obj
    p = Path(path_or_obj)
    if not p.exists():
        raise FileNotFoundError(f"TriviaQA dataset file not found: {path_or_obj}")
    with open(p, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    return data


def get_key_to_ground_truth(dataset_json: Dict[str, Any]):
    """
    Convert the official TriviaQA dataset JSON into a mapping from question-id -> ground-truth object.
    The official evaluator expects the structure it uses (NormalizedAliases and HumanAnswers).
    If the dataset JSON is already in the expected format, return as-is.
    """
    # If it already contains the top-level 'Data' or 'Version' keys, assume it's the official dump
    if isinstance(dataset_json, dict) and ('Data' in dataset_json or 'questions' in dataset_json):
        # Best-effort conversion: if 'Data' exists, flatten it
        if 'Data' in dataset_json:
            mapping = {}
            for entry in dataset_json['Data']:
                # official dumps sometimes have 'QuestionId' or 'QuestionId' nested
                qid = entry.get('QuestionId') or entry.get('QuestionId') or entry.get('question_id') or entry.get('Question')
                if qid is None:
                    continue
                mapping[str(qid)] = entry
            return mapping
        # else return as-is if keys look like qid -> obj
        # If top-level looks already like key->ground truth mapping
        if all(isinstance(k, str) for k in dataset_json.keys()):
            return dataset_json

    # As a fallback, if given a list of QA examples, map index->example
    if isinstance(dataset_json, list):
        return {str(i): v for i, v in enumerate(dataset_json)}

    # Last resort: return as-is
    return dataset_json
