#!/usr/bin/env python3
"""
Evaluate a Self-RAG model on a Hugging Face dataset split (e.g., RAGTruth) via a TGI endpoint.

Requirements:
  pip install datasets requests tqdm numpy

Example usage:
  python evals/ragtruth_hf_eval.py \
    --tgi-url http://localhost:8300 \
    --dataset-name RAGTruth/ragtruth \
    --split validation \
    --output-file outputs/ragtruth_preds.jsonl

If your columns differ, specify:
  --input-column prompt --context-column context --label-column label
or let the script auto-detect from common names:
  - prompt: prompt | question | query | input | instruction
  - context: context | contexts | passages | docs | evidence | source_documents
  - label: label | hallucinated | hallucination | is_hallucination | gold_label

Label mapping:
  - Any of yes,true,1 are treated as positive ('yes')
  - Any of no,false,0 are treated as negative ('no')
  Override with --positive-label-values and --negative-label-values if needed.
"""

import argparse, json, os, re, time
from typing import Any, Dict, List, Optional
import requests
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgi-url", type=str, default="http://localhost:8300", help="Base URL of TGI server")
    ap.add_argument("--dataset-name", type=str, required=True, help="HF dataset name or path (e.g., RAGTruth/ragtruth)")
    ap.add_argument("--dataset-config", type=str, default=None, help="HF dataset config name (optional)")
    ap.add_argument("--split", type=str, default="validation", help="Split name (e.g., validation, test)")
    ap.add_argument("--output-file", type=str, default="outputs/ragtruth_preds.jsonl")
    ap.add_argument("--system-prompt", type=str, default="", help="Optional system prompt prefix")

    # Column hints; if omitted, we try to auto-detect.
    ap.add_argument("--input-column", type=str, default=None, help="Column containing the main user question/prompt")
    ap.add_argument("--context-column", type=str, default=None, help="Column containing retrieval/context (str or list[str])")
    ap.add_argument("--label-column", type=str, default=None, help="Column containing ground-truth yes/no or 0/1")

    # Label mapping controls
    ap.add_argument("--positive-label-values", type=str, default="yes,true,1", help="Comma-separated values considered positive")
    ap.add_argument("--negative-label-values", type=str, default="no,false,0", help="Comma-separated values considered negative")

    # Generation params
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    return ap.parse_args()

def normalize_label(x: Any, pos_vals: set, neg_vals: set) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in pos_vals:
        return "yes"
    if s in neg_vals:
        return "no"
    # Try to coerce ints/bools
    if s.isdigit():
        return "yes" if int(s) != 0 else "no"
    if s in {"true"}:
        return "yes"
    if s in {"false"}:
        return "no"
    return None

def extract_yesno(text: str) -> Optional[str]:
    if not text:
        return None
    m = YESNO_RE.search(text)
    return m.group(1).lower() if m else None

def auto_pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def build_prompt(question: str, context: Optional[Any], system_prompt: str) -> str:
    ctx = ""
    if context is not None:
        if isinstance(context, list):
            ctx = "\n\n".join(map(str, context))
        else:
            ctx = str(context)
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    parts.append(f"Question:\n{question}")
    if ctx:
        parts.append(f"Context:\n{ctx}")
    parts.append("Answer strictly 'yes' or 'no' to whether the response contains unsupported claims.")
    return "\n\n".join(parts)

def tgi_generate(tgi_url: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop": [],
            "do_sample": temperature > 0.0
        }
    }
    for _ in range(3):
        r = requests.post(f"{tgi_url}/generate", json=payload, timeout=120)
        if r.status_code == 200:
            data = r.json()
            return data.get("generated_text", "")
        time.sleep(1.0)
    raise RuntimeError(f"TGI generate failed after retries (url={tgi_url}).")

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Load HF dataset split
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

    cols = list(ds.features.keys())
    input_col = args.input_column or auto_pick(cols, ["prompt", "question", "query", "input", "instruction"])
    context_col = args.context_column or auto_pick(cols, ["context", "contexts", "passages", "docs", "evidence", "source_documents"])
    label_col = args.label_column or auto_pick(cols, ["label", "hallucinated", "hallucination", "is_hallucination", "gold_label", "target"])

    if not input_col:
        raise ValueError(f"Could not auto-detect an input column. Available columns: {cols}. Provide --input-column.")
    # label_col is optional; if not found we'll just write predictions

    pos_vals = {s.strip().lower() for s in args.positive_label_values.split(",") if s.strip()}
    neg_vals = {s.strip().lower() for s in args.negative_label_values.split(",") if s.strip()}

    preds_bin: List[int] = []
    gts_bin: List[int] = []

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for ex in tqdm(ds, desc="Evaluating", unit="ex"):
            question = ex.get(input_col)
            context = ex.get(context_col) if context_col else None
            label_raw = ex.get(label_col) if label_col else None
            gt = normalize_label(label_raw, pos_vals, neg_vals)

            prompt = build_prompt(str(question), context, args.system_prompt)
            gen = tgi_generate(args.tgi_url, prompt, args.temperature, args.top_p, args.max_new_tokens)
            pred = extract_yesno(gen) or extract_yesno(gen.splitlines()[0] if gen else "") or "no"

            rec = {
                "prompt": question,
                "context": context,
                "prediction": pred,
                "label": gt,
                "raw_output": gen
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if gt is not None:
                preds_bin.append(1 if pred == "yes" else 0)
                gts_bin.append(1 if gt == "yes" else 0)

    if gts_bin:
        y_true = np.array(gts_bin)
        y_pred = np.array(preds_bin)
        acc = (y_true == y_pred).mean().item()
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics = {
            "accuracy": round(float(acc), 6),
            "precision_yes": round(float(prec), 6),
            "recall_yes": round(float(rec), 6),
            "f1_yes": round(float(f1), 6),
            "support": int(len(y_true))
        }
        print("\nMetrics:", json.dumps(metrics, indent=2))
        with open(os.path.splitext(args.output_file)[0] + ".metrics.json", "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2)
    else:
        print("\nNo ground-truth labels found. Wrote predictions only to:", args.output_file)

if __name__ == "__main__":
    main()
