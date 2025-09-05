#!/usr/bin/env python3
"""
run_eval.py

Research-ready evaluation runner for SELF-RAG inference only, aligned with the Self-RAG README (7B model).

Usage:
    python run_eval.py \
      --model_name selfrag/selfrag_llama2_7b \
      --n_samples 200 \
      --max_new_tokens 512 \
      --output_dir ./selfrag_eval_outputs

Notes:
 - Requires: vllm, datasets, python-standard libs.
 - Log in to HuggingFace first if datasets require auth (`huggingface-cli login`).
 - For RAGTruth span-level metrics, this script currently does not compute span-level PRF (left as None).
 - EM is improved to account for surface wording differences using normalization, containment, and fuzzy ratio.
 - Follows the Self-RAG README formatting and defaults for vLLM: dtype default is "half" and skip_special_tokens=False.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import subprocess

# ---------------- Dependencies ----------------
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise SystemExit(
        "vllm is required but not importable. Install with `pip install vllm` and ensure GPU drivers are set up. Error: {}".format(e)
    )

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("datasets is required. Install with `pip install datasets`. Error: {}".format(e))

# Optional: Hugging Face Hub login for private/authorized models
try:
    from huggingface_hub import login as hf_login, HfFolder
except Exception:
    hf_login = None
    HfFolder = None

# ---------------- Constants & thresholds ----------------
FUZZY_EM_RATIO = 0.92  # SequenceMatcher ratio threshold for fuzzy EM (conservative)
DEFAULT_BATCH_SIZE = 4
# Keep 'half' as the default to follow README, but coerce internally to a valid vLLM dtype.
DEFAULT_DTYPE = "half"

# ---------------- Normalization & scoring utilities ----------------
def normalize_answer(s: Optional[str]) -> str:
    """Normalize text: lower, remove punctuation, articles, extra spaces."""
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    # remove punctuation (keep word characters and spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def exact_match_improved(pred: str, gold: str) -> int:
    """Improved EM:
       - normalized equality OR
       - normalized containment (pred in gold or gold in pred) OR
       - fuzzy ratio >= FUZZY_EM_RATIO
    """
    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)
    if not npred and not ngold:
        return 1
    if npred == ngold:
        return 1
    # containment: handles "obama" vs "barack obama"
    if npred in ngold or ngold in npred:
        return 1
    # fuzzy edit-similarity guard (conservative)
    if fuzzy_ratio(npred, ngold) >= FUZZY_EM_RATIO:
        return 1
    return 0

def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (standard)."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p_tokens)
    rec = same / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

def lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
    """Length of the Longest Common Subsequence between two token lists (O(n*m))."""
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[n][m]

def rouge_l_f1(pred: str, gold: str) -> float:
    """Compute ROUGE-L F1 using LCS over normalized tokens."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    lcs = lcs_length(p_tokens, g_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_tokens)
    rec = lcs / len(g_tokens)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def best_rouge_l_over_golds(pred: str, golds: Optional[List[str]]) -> float:
    if not golds:
        return 0.0
    best = 0.0
    for g in golds:
        score = rouge_l_f1(pred, g)
        if score > best:
            best = score
    return best

def best_em_f1_over_golds(pred: str, golds: Optional[List[str]]) -> Tuple[int, float]:
    """Return best EM and best F1 across multiple gold answers"""
    if not golds:
        return 0, 0.0
    best_em = 0
    best_f1 = 0.0
    for g in golds:
        em = exact_match_improved(pred, g)
        f1 = token_f1(pred, g)
        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
    return best_em, best_f1

def inclusion_match(pred: str, golds: Optional[List[str]]) -> int:
    """Return 1 if any normalized gold string is contained in normalized pred."""
    if not golds:
        return 0
    npred = normalize_answer(pred)
    for g in golds:
        if normalize_answer(g) and normalize_answer(g) in npred:
            return 1
    return 0

# ---------------- RAGTruth helpers (response-level & span-level) ----------------
def detect_ragtruth_response_label(example: dict) -> Optional[int]:
    """
    Heuristic: try to extract a binary label for hallucination existence.
    Return 1 if example marked hallucinated (positive), 0 if not hallucinated, None if unknown.
    """
    # common possible fields
    fields = ["is_hallucinated", "hallucinated", "label", "gold_label", "y", "has_hallucination"]
    for f in fields:
        if f in example:
            val = example[f]
            # numeric or boolean
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return int(bool(val))
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("yes", "true", "1", "hallucinated", "hallucination", "halluc"):
                    return 1
                if v in ("no", "false", "0", "not enough", "not_hallucinated", "non-hallucinated", "non-halluc"):
                    return 0
    # sometimes stored under nested annotations (try find any 'halluc' token)
    for k, v in example.items():
        if "halluc" in str(k).lower():
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, (int, float)):
                return int(bool(v))
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv in ("yes", "true", "1"):
                    return 1
                if vv in ("no", "false", "0"):
                    return 0
    return None

def extract_ragtruth_spans(example: dict) -> Optional[List[Tuple[int,int]]]:
    """
    Best-effort: find annotated hallucinated spans. Return list of (start_char, end_char) spans.
    If dataset stores token indices or char spans, try to convert.
    """
    # try common keys
    span_keys = ["hallucinated_spans", "spans", "annotations", "halluc_spans"]
    for k in span_keys:
        if k in example and example[k]:
            val = example[k]
            # If list of dicts with start/end
            if isinstance(val, list):
                spans = []
                for item in val:
                    if isinstance(item, dict):
                        # common keys: start, end, s, e
                        start = item.get("start") or item.get("s") or item.get("char_start")
                        end = item.get("end") or item.get("e") or item.get("char_end")
                        if start is not None and end is not None:
                            try:
                                spans.append((int(start), int(end)))
                            except Exception:
                                pass
                if spans:
                    return spans
            # If list of (start,end) tuples serialized as lists
            if isinstance(val, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in val):
                try:
                    return [(int(x[0]), int(x[1])) for x in val]
                except Exception:
                    pass
    return None

def compute_span_level_prf(pred_text: str, gold_spans: List[Tuple[int,int]]) -> Optional[Tuple[float,float,float]]:
    """
    Placeholder for span-level PRF computation.

    Real span-level evaluation requires alignment between gold spans and prediction text.
    This function currently returns None to indicate 'not computed'.
    """
    return None

# ---------------- Dataset extraction heuristics ----------------
def extract_fields_for_dataset(example: dict, dataset: str) -> Tuple[Optional[str], Optional[List[str]], Optional[str], dict]:
    """
    Return (instruction_text, list_of_gold_answers, optional_context_paragraph, metadata_dict)
    metadata_dict contains raw fields used to help later (e.g., labels for fever, ragtruth info)
    """
    # Generic candidates
    q_candidates = ["question", "query", "claim", "instruction", "input", "query_text", "question_text", "question_body"]
    ans_candidates = ["answers", "answer", "gold_answers", "label", "answer_text", "target_text"]
    context_candidates = ["contexts", "context", "retrieved_docs", "paragraph", "evidence", "document", "context_text", "passage"]
    meta = {}
    instruction = None

    # direct key matches preferred
    for k in q_candidates:
        if k in example and example[k]:
            instruction = example[k]
            break

    # dataset-specific fallbacks
    if instruction is None:
        if dataset.startswith("mwong/fever"):
            if "claim" in example:
                instruction = example["claim"]
        if dataset.startswith("mandarjoshi/trivia_qa"):
            if "question" in example:
                instruction = example["question"]
        if dataset.startswith("microsoft/ms_marco"):
            if "query" in example:
                instruction = example["query"]
    # Answers extraction
    answers = None
    if "answers" in example:
        a = example["answers"]
        if isinstance(a, dict):
            # many HF qa formats: {'text': [...]} or {'answer': ...}
            if "text" in a:
                answers = a["text"] if isinstance(a["text"], list) else [a["text"]]
            elif "answer" in a:
                answers = a["answer"] if isinstance(a["answer"], list) else [a["answer"]]
        elif isinstance(a, list):
            answers = a
        elif isinstance(a, str):
            answers = [a]
    else:
        for k in ans_candidates:
            if k in example and example[k]:
                v = example[k]
                if isinstance(v, list):
                    answers = v
                elif isinstance(v, dict):
                    if "text" in v and isinstance(v["text"], list):
                        answers = v["text"]
                    else:
                        answers = [str(x) for x in v.values() if x]
                else:
                    answers = [str(v)]
                break

    # TriviaQA-specific: gold answers live under 'answer' dict with 'value' and 'aliases'
    if dataset.startswith("mandarjoshi/trivia_qa"):
        try:
            a = example.get("answer") if isinstance(example, dict) else None
            if isinstance(a, dict):
                golds: List[str] = []
                val = a.get("value")
                if isinstance(val, str) and val.strip():
                    golds.append(val)
                aliases = a.get("aliases")
                if isinstance(aliases, list):
                    golds.extend([str(x) for x in aliases if isinstance(x, str) and x.strip()])
                if golds:
                    answers = golds
        except Exception:
            pass

    # context
    context = None
    for k in context_candidates:
        if k in example and example[k]:
            context = example[k]
            break

    # normalize answers to list[str] or None
    if answers is not None:
        clean = []
        for a in answers:
            if a is None:
                continue
            if isinstance(a, (list, dict)):
                clean.append(str(a))
            else:
                clean.append(str(a))
        answers = clean if clean else None

    # metadata for FEVER labels and RAGTruth
    if "label" in example:
        meta["label"] = example["label"]
    if "gold_label" in example:
        meta["gold_label"] = example["gold_label"]
    # RAGTruth detection fields
    meta["ragtruth_label"] = detect_ragtruth_response_label(example)
    meta["ragtruth_spans"] = extract_ragtruth_spans(example)

    return (str(instruction) if instruction else None, answers, (str(context) if context else None), meta)

# ---------------- vllm helpers ----------------
def get_text_from_vllm_result(pred_obj) -> str:
    """Safely extract text from a vllm result item."""
    try:
        # vllm generate returns objects with .outputs; each output is a list of result alternatives with .text
        if hasattr(pred_obj, "outputs"):
            out0 = pred_obj.outputs[0]
            if hasattr(out0, "text"):
                return out0.text.strip()
            return str(out0).strip()
        # fallback
        return str(pred_obj).strip()
    except Exception:
        try:
            return str(pred_obj).strip()
        except Exception:
            return ""

def batched_generate(model: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """
    Generate for a list of prompts using vllm. Returns list of strings (model outputs).
    If batch generation fails, fallback to single-call generation per prompt.
    """
    if not prompts:
        return []
    # Try batched generation
    try:
        preds = model.generate(prompts, sampling_params)
        # model.generate returns a list-like object; extract text for each
        texts = []
        for p in preds:
            texts.append(get_text_from_vllm_result(p))
        return texts
    except Exception as e:
        print("vllm batched generation failed (will fallback to single-call loop). Error:", e, file=sys.stderr)
        texts = []
        for p in prompts:
            try:
                single = model.generate([p], sampling_params)[0]
                texts.append(get_text_from_vllm_result(single))
            except Exception as e2:
                print("vllm single-call generation failed for one prompt:", e2, file=sys.stderr)
                texts.append("")
        return texts

# ---------------- Core evaluation per dataset (batched) ----------------
def _find_label_from_example(ex: dict, meta: dict) -> Optional[str]:
    """Look for FEVER label in meta and the raw example under many common keys."""
    # priority: meta if present
    candidates = []
    if isinstance(meta, dict):
        candidates.extend(["label", "gold_label", "ragtruth_label"])
        for k in ["label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label"]:
            if k in meta:
                candidates.append(k)
    # also check the raw example
    candidates.extend(["label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label", "verifiable"])
    # finally any key that looks promising
    for k in list(ex.keys()):
        if k.lower() in ("label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label"):
            candidates.append(k)
    # dedupe preserve order
    seen = set()
    final_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            final_candidates.append(c)
    # check them
    for k in final_candidates:
        if isinstance(meta, dict) and k in meta and meta[k] is not None:
            return meta[k]
        if k in ex and ex[k] is not None:
            return ex[k]
    # fallback: some datasets store label under 'annotations' or nested dicts
    for k, v in ex.items():
        if isinstance(k, str) and "label" in k.lower():
            return v
    return None

def evaluate_fever(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """FEVER: classification accuracy (kept exactly as before) plus evidence metrics and FEVER score.

    Notes:
    - This implementation preserves the original accuracy calculation and which examples are counted
      (i.e., the same 'correct' and 'evaluated' logic as the prior version).
    - Additional outputs added: per-example predicted evidence, evidence P/R/F1, whether any gold evidence set
      was covered, aggregated micro/macro evidence metrics, per-label P/R/F1, and FEVER score.
    """
    print("Evaluating FEVER (accuracy + evidence metrics)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    correct = 0
    latencies = []
    label_mode_used = None  # '3way' or 'binary'

    # Helpers for evidence extraction and matching
    def _get_gold_evidence_sets(ex: dict) -> Optional[List[set]]:
        """Return list of gold-evidence-sets, each set is a set of (doc_id, sent_idx) tuples.
        Parse common FEVER formats: lists of lists, lists of dicts with doc/sent keys, tuples, or strings like 'Title#3'.
        """
        # Prefer strict FEVER v2.0 style fields when available (this yields precise doc+sent tuples)
        # FEVER v2.0 generator exposes fields like: evidence_annotation_id, evidence_sentence_id,
        # evidence_wiki_url, evidence_id. If present, create a single gold set per example entry.
        if isinstance(ex, dict) and ("evidence_annotation_id" in ex or "evidence_sentence_id" in ex or "evidence_wiki_url" in ex or "evidence_id" in ex):
            try:
                annot = ex.get("evidence_annotation_id")
                sent = ex.get("evidence_sentence_id")
                wiki = ex.get("evidence_wiki_url") or ex.get("evidence_wiki") or ex.get("evidence_wikipedia_url")
                eid = ex.get("evidence_id")
                # If sentence id is present and wiki/page id or url present, return that as the only gold set
                if sent is not None and (wiki or eid is not None or annot is not None):
                    doc_id = str(wiki) if wiki else (str(eid) if eid is not None else str(annot))
                    try:
                        sent_idx = int(sent)
                    except Exception:
                        try:
                            sent_idx = int(float(sent))
                        except Exception:
                            sent_idx = None
                    if sent_idx is not None:
                        return [set([(doc_id.strip(), sent_idx)])]
            except Exception:
                # fall back to heuristics below
                pass

        candidates = []
        for k in ("evidence", "evidence_sets", "evidences", "annotated_evidence", "evidence_list", "evidence_sentences"):
            if k in ex and ex[k]:
                candidates.append(ex[k])

        if not candidates:
            return None

        def _parse_item(item) -> Optional[Tuple[str,int]]:
            # dict-like
            if isinstance(item, dict):
                doc_keys = ["doc_id", "page", "wiki_id", "wikipedia_id", "document", "doc", "page_title", "title"]
                sent_keys = ["sentence_id", "sent_id", "sentence_index", "sent_index", "s", "sentence"]
                doc = None
                sent = None
                for k in doc_keys:
                    if k in item and item[k] is not None:
                        doc = str(item[k])
                        break
                for k in sent_keys:
                    if k in item and item[k] is not None:
                        try:
                            sent = int(item[k])
                        except Exception:
                            try:
                                sent = int(float(item[k]))
                            except Exception:
                                sent = None
                        break
                if doc is not None and sent is not None:
                    return (doc.strip(), int(sent))

            # list/tuple of length 2
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    doc = str(item[0]).strip()
                    sent = int(item[1])
                    return (doc, sent)
                except Exception:
                    pass

            # string patterns like Title#3 or (Title, 3)
            if isinstance(item, str):
                s = item.strip()
                m = re.search(r"^\s*['\"]?(?P<doc>.+?)['\"]?\s*#\s*(?P<sent>\d+)\s*$", s)
                if m:
                    return (m.group("doc").strip(), int(m.group("sent")))
                m = re.search(r"\((?P<doc>[^,]+),\s*(?P<sent>\d+)\)", s)
                if m:
                    return (m.group("doc").strip(), int(m.group("sent")))
            return None

        gold_sets: List[set] = []
        for cand in candidates:
            # case: list of evidence-sets (each set is a list)
            if isinstance(cand, list) and cand and isinstance(cand[0], list):
                for inner in cand:
                    cur = set()
                    for it in inner:
                        parsed = _parse_item(it)
                        if parsed:
                            cur.add(parsed)
                    if cur:
                        gold_sets.append(cur)
            # case: single list of evidence items (treat as one set)
            elif isinstance(cand, list):
                cur = set()
                for it in cand:
                    parsed = _parse_item(it)
                    if parsed:
                        cur.add(parsed)
                if cur:
                    gold_sets.append(cur)
            # case: dict mapping or single dict item
            elif isinstance(cand, dict):
                for v in cand.values():
                    if isinstance(v, (list, tuple)):
                        cur = set()
                        for it in v:
                            parsed = _parse_item(it)
                            if parsed:
                                cur.add(parsed)
                        if cur:
                            gold_sets.append(cur)
        return gold_sets if gold_sets else None

    def _parse_predicted_evidence_from_text(text: str) -> List[Tuple[str,int]]:
        """Heuristic parse predicted evidence from model text into a list of (doc, sent) tuples.
        We look for patterns like Title#3, (Title, 3), ['Title', 3], doc: Title, sent: 3, etc.
        Returns list of tuples.
        """
        if not text:
            return []
        t = text.strip()
        # strip predicted label prefix
        lab_pat = re.search(r"\b(SUPPORTS|REFUTES|NOT ENOUGH INFO|NOT ENOUGH|NEI)\b", t, flags=re.IGNORECASE)
        tail = t[lab_pat.end():].strip() if lab_pat else t
        # look for common evidence markers
        m = re.search(r"evidence[:\s]*", tail, flags=re.IGNORECASE)
        if m:
            tail = tail[m.end():].strip()

        preds = set()
        # pattern: Title#3
        for m in re.finditer(r"([A-Za-z0-9 _\-/]+?)#(\d+)", tail):
            doc = m.group(1).strip()
            sent = int(m.group(2))
            preds.add((doc, sent))
        # pattern: (Title, 3)
        for m in re.finditer(r"\(([^,()]+),\s*(\d+)\)", tail):
            doc = m.group(1).strip().strip('"\'')
            sent = int(m.group(2))
            preds.add((doc, sent))
        # pattern: ['Title', 3] or ["Title", 3]
        for m in re.finditer(r"\[\s*['\"]([^'\"]+)['\"]\s*,\s*(\d+)\s*\]", tail):
            doc = m.group(1).strip()
            sent = int(m.group(2))
            preds.add((doc, sent))
        # pattern: doc: Title; sent: 3 or document: Title, sentence: 3
        for m in re.finditer(r"doc(?:ument)?[:=]\s*([^;,\n]+)[,;\n]+\s*sent(?:ence)?[:=]\s*(\d+)", tail, flags=re.IGNORECASE):
            doc = m.group(1).strip()
            sent = int(m.group(2))
            preds.add((doc, sent))

        return list(preds)

    # Aggregators for evidence micro metrics
    micro_tp = 0  # total matched evidence instances
    micro_pred = 0  # total predicted evidence instances
    micro_gold = 0  # total gold evidence instances
    per_label_counts: Dict[str, Dict[str, int]] = {}  # label -> {'tp':..,'pred':..,'gold':..,'examples':..}
    fever_both_correct = 0

    # batch processing (preserve original accuracy logic)
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        prompt_meta = []  # (idx, ex, meta, instruction)

        # build prompts for this batch (keep original prompt text to preserve accuracy behavior)
        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, "mwong/fever-evidence-related")
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-claim"})
                continue
            prompt = format_prompt(f"Verify this claim: {instruction}\nAnswer with one of: SUPPORTS, REFUTES, NOT ENOUGH INFO.")
            actual_prompts.append(prompt)
            prompt_meta.append((idx, ex, meta, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, ex, meta, instruction), text in zip(prompt_meta, texts):
            mapped = map_to_fever_label(text)
            correct_flag = 0
            gold_label = _find_label_from_example(ex, meta)
            binary_gold = None
            mode = None

            # Determine if the found gold label is numeric/binary (preserve old logic structure)
            def as_binary(val):
                try:
                    if isinstance(val, bool):
                        return 1 if val else 0
                    if isinstance(val, (int, float)):
                        return int(bool(val))
                    if isinstance(val, str):
                        v = str(val).strip().lower()
                        if v in ("1", "true", "yes", "support", "supports"):
                            return 1
                        if v in ("0", "false", "no", "refute", "refutes"):
                            return 0
                except Exception:
                    return None
                return None

            if gold_label is not None:
                maybe_bin = as_binary(gold_label)
                if maybe_bin is not None:
                    binary_gold = maybe_bin
                    mode = "binary"
                else:
                    mode = "3way"
            else:
                if isinstance(ex, dict) and "labels" in ex:
                    lb = ex.get("labels")
                    maybe_bin = as_binary(lb)
                    if maybe_bin is not None:
                        binary_gold = maybe_bin
                        mode = "binary"

            # determine correctness for label (exact same calculation as before)
            if gold_label is not None:
                if normalize_label_compare(mapped, gold_label):
                    correct_flag = 1
                else:
                    correct_flag = 0
            elif binary_gold is not None:
                # In binary mode, map SUPPORTS->1, REFUTES->0, NEI -> None
                mapped_bin = None
                if mapped == "SUPPORTS":
                    mapped_bin = 1
                elif mapped == "REFUTES":
                    mapped_bin = 0
                if mapped_bin is not None and binary_gold is not None:
                    correct_flag = 1 if mapped_bin == binary_gold else 0
                else:
                    correct_flag = 0

            if mode and (label_mode_used is None):
                label_mode_used = mode

            # Evidence handling: extract gold sets and predicted evidence (heuristic)
            gold_sets = _get_gold_evidence_sets(ex)  # list of lists (normalized strings)
            pred_evidence = _parse_predicted_evidence_from_text(text)

            # Compute per-example evidence PRF per FEVER spec (flattened sets)
            evidence_p = evidence_r = evidence_f = None
            evidence_covered = False
            # predicted evidence parsed into list of (doc,sent) tuples
            pred_tuples = set(pred_evidence) if pred_evidence else set()
            # gold_sets is a list of sets of (doc,sent)
            if gold_sets:
                # flattened gold sentences
                try:
                    gold_union = set().union(*gold_sets)
                except Exception:
                    gold_union = set()
                tp = len(pred_tuples & gold_union)
                pred_count = len(pred_tuples)
                gold_count = len(gold_union)
                evidence_p = (tp / pred_count) if pred_count > 0 else 0.0
                evidence_r = (tp / gold_count) if gold_count > 0 else 0.0
                evidence_f = (2 * evidence_p * evidence_r / (evidence_p + evidence_r)) if (evidence_p + evidence_r) > 0 else 0.0
                # full-coverage: predicted sentences superset of any gold set
                try:
                    evidence_covered = any(pred_tuples.issuperset(gset) for gset in gold_sets)
                except Exception:
                    evidence_covered = False
                # update micro accumulators (flattened as spec)
                micro_tp += tp
                micro_pred += pred_count
                micro_gold += gold_count
            else:
                # no gold evidence available (e.g., NEI); per-spec treat precision/recall as 0 when denom 0
                tp = 0
                pred_count = len(pred_tuples)
                gold_count = 0
                evidence_p = 0.0 if pred_count > 0 else 0.0
                evidence_r = 0.0
                evidence_f = 0.0
                micro_tp += 0
                micro_pred += pred_count
                micro_gold += 0

            # FEVER score: per-spec (NEI handled specially)
            fever_score_flag = False
            gold_is_nei = False
            try:
                if isinstance(gold_label, str) and normalize_answer(str(gold_label)).startswith("not enough"):
                    gold_is_nei = True
            except Exception:
                gold_is_nei = False

            if gold_is_nei:
                if correct_flag:
                    fever_score_flag = True
                    fever_both_correct += 1
            else:
                if correct_flag and evidence_covered:
                    fever_score_flag = True
                    fever_both_correct += 1

            outputs.append({
                "index": idx,
                "instruction": instruction,
                "model_raw": text,
                "pred_label": mapped,
                "gold_label": gold_label,
                "binary_gold": binary_gold,
                "mode_used": mode,
                "correct": correct_flag,
                "pred_evidence": pred_evidence,
                "gold_evidence_sets": gold_sets,
                "evidence_precision": evidence_p,
                "evidence_recall": evidence_r,
                "evidence_f1": evidence_f,
                "evidence_covers_gold_set": evidence_covered,
                "fever_score": fever_score_flag,
                "latency_s": per_latency
            })

            # update per-label counters for micro/macro using flattened counts
            lbl = str(gold_label) if gold_label is not None else ("binary_" + str(binary_gold) if binary_gold is not None else "unknown")
            if lbl not in per_label_counts:
                per_label_counts[lbl] = {"tp": 0, "pred": 0, "gold": 0, "examples": 0, "f_sum": 0.0}
            per_label_counts[lbl]["pred"] += pred_count
            per_label_counts[lbl]["tp"] += tp
            per_label_counts[lbl]["gold"] += gold_count
            per_label_counts[lbl]["examples"] += 1
            per_label_counts[lbl]["f_sum"] += (evidence_f if evidence_f is not None else 0.0)

            correct += correct_flag

    # Compute FEVER metrics using robust set-based logic (user-provided implementation)
    def normalize_label(label):
        mapping = {
            "SUPPORTS": "SUPPORTS",
            "SUPPORTED": "SUPPORTS",
            "REFUTES": "REFUTES",
            "REFUTED": "REFUTES",
            "NOT ENOUGH INFO": "NOT ENOUGH INFO",
            "NEI": "NOT ENOUGH INFO",
            "0": "REFUTES",
            "1": "SUPPORTS",
        }
        if label is None:
            return None
        s = str(label).strip().upper()
        return mapping.get(s, s)

    def evidence_set_for_gold_item(o: dict) -> List[set]:
        # Expect o["gold_evidence_sets"] to be list of sets of (doc,sent) tuples
        g = o.get("gold_evidence_sets")
        sets = []
        if g:
            # if it's a list of sets or list of lists
            for group in g:
                cur = set()
                try:
                    for entry in group:
                        # entry should be tuple (doc, sent)
                        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            wiki = entry[0]
                            sent = entry[1]
                            try:
                                senti = int(sent)
                            except Exception:
                                try:
                                    senti = int(float(sent))
                                except Exception:
                                    senti = -1
                            cur.add((str(wiki), senti))
                except Exception:
                    continue
                if cur:
                    sets.append(cur)
        # If no candidate sets, return list with empty set (per user code)
        return sets or [set()]

    def evidence_set_for_pred_item(pred_list) -> set:
        preds = set()
        if not pred_list:
            return preds
        for x in pred_list:
            try:
                doc = str(x[0])
                sent = int(x[1])
                preds.add((doc, sent))
            except Exception:
                # skip malformed
                pass
        return preds

    label_acc = 0
    fever_score_count = 0
    p_micro_sum = 0.0
    r_micro_sum = 0.0
    f1_micro_sum = 0.0
    total = len([o for o in outputs if not o.get("skipped", False)])

    for o in outputs:
        if o.get("skipped", False):
            continue
        gold_label_raw = o.get("gold_label") if o.get("gold_label") is not None else o.get("binary_gold")
        gold_label_norm = normalize_label(gold_label_raw) if gold_label_raw is not None else None
        pred_label_norm = normalize_label(o.get("pred_label")) if o.get("pred_label") is not None else None

        # Label accuracy
        if pred_label_norm is not None and gold_label_norm is not None and pred_label_norm == gold_label_norm:
            label_acc += 1

        # Evidence sets and prediction
        gold_sets = evidence_set_for_gold_item(o)
        pred_evidence = evidence_set_for_pred_item(o.get("pred_evidence", []))

        # FEVER score: label correct and predicted superset of any gold set OR gold is NEI
        full_coverage = any((len(s) == 0 and len(pred_evidence) == 0) or (s and s.issubset(pred_evidence)) for s in gold_sets)
        if (pred_label_norm is not None and gold_label_norm is not None and pred_label_norm == gold_label_norm) and (full_coverage or gold_label_norm == "NOT ENOUGH INFO"):
            fever_score_count += 1

        # Micro evidence metrics: flatten all gold sets
        gold_evidence_flat = set()
        for s in gold_sets:
            gold_evidence_flat |= s

        if pred_evidence:
            prec = len(pred_evidence & gold_evidence_flat) / len(pred_evidence)
        else:
            prec = 0.0
        if gold_evidence_flat:
            rec = len(pred_evidence & gold_evidence_flat) / len(gold_evidence_flat)
        else:
            rec = 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        p_micro_sum += prec
        r_micro_sum += rec
        f1_micro_sum += f1

    # Avoid division by zero
    if total == 0:
        label_accuracy = None
        fever_score = None
        evidence_precision_micro = None
        evidence_recall_micro = None
        evidence_f1_micro = None
    else:
        label_accuracy = label_acc / total
        fever_score = fever_score_count / total
        evidence_precision_micro = p_micro_sum / total
        evidence_recall_micro = r_micro_sum / total
        evidence_f1_micro = f1_micro_sum / total

    out_file = out_dir / "fever_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for item in outputs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "dataset": "mwong/fever-evidence-related",
        "split": split,
        "examples_evaluated": total,
        "label_accuracy": label_accuracy,
        "fever_score": fever_score,
        "evidence_precision_micro": evidence_precision_micro,
        "evidence_recall_micro": evidence_recall_micro,
        "evidence_f1_micro": evidence_f1_micro,
        "output_file": str(out_file),
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None
    }

def map_to_fever_label(text: str) -> str:
    """Map arbitrary model output to one of SUPPORTS / REFUTES / NOT ENOUGH INFO (NEI first; conservative, word-boundary based)."""
    t = normalize_answer(text)

    # Handle explicit 'not support' forms as REFUTES early
    if re.search(r"\b(not|no)\s+(support|supported|supports)\b", t):
        return "REFUTES"

    # NOT ENOUGH INFO (NEI) first
    if (
        "not enough" in t
        or "insufficient" in t
        or "no evidence" in t
        or "unknown" in t
        or "cannot determine" in t
        or "unable to determine" in t
        or "cannot tell" in t
        or "indeterminate" in t
        or re.search(r"\bnei\b", t) is not None
    ):
        return "NOT ENOUGH INFO"

    # REFUTES
    if (
        re.search(r"\brefut\w*\b", t) is not None
        or re.search(r"\bcontradict\w*\b", t) is not None
        or re.search(r"\bdisprov\w*\b", t) is not None
        or re.search(r"(?<!not\s)\bfalse\b", t) is not None
        or re.search(r"\bdoes\s+not\s+(hold|follow)\b", t) is not None
    ):
        return "REFUTES"

    # SUPPORTS
    if (
        re.search(r"(?<!not\s)\b(support|supports|supported|entail|entails|entailed)\b", t) is not None
        or re.search(r"(?<!not\s)\b(true|yes)\b", t) is not None
    ):
        return "SUPPORTS"

    # Default to NEI
    return "NOT ENOUGH INFO"

def normalize_label_compare(pred_label: Optional[str], gold_label: Optional[str]) -> bool:
    if pred_label is None or gold_label is None:
        return False

    def canonize(x: str) -> str:
        x = normalize_answer(x)
        # Map synonyms and variants to canonical set
        mapping = {
            "supported": "supports",
            "support": "supports",
            "entail": "supports",
            "entails": "supports",
            "entailed": "supports",
            "refuted": "refutes",
            "refute": "refutes",
            "contradiction": "refutes",
            "contradicts": "refutes",
            "nei": "not enough info",
        }
        return mapping.get(x, x)

    pl = canonize(pred_label)
    gl = canonize(gold_label)
    # Golds are commonly "SUPPORTS"/"REFUTES"/"NOT ENOUGH INFO"
    return pl == gl

def evaluate_generic_qa(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int, dataset_name: str, metric_mode: str):
    """
    Generic QA evaluation for MSMARCO, HotPotQA, TriviaQA, Natural Questions.
    metric_mode: 'em_f1' (EM+F1), 'inclusion' (match/inclusion)
    """
    print(f"Evaluating {dataset_name} with mode {metric_mode}")
    split = choose_split(ds)
    data = ds[split]
    # Support both in-memory datasets and streaming IterableDataset
    supports_index = hasattr(data, "__len__") and hasattr(data, "__getitem__")
    data_buffer = None
    if supports_index:
        total = len(data)
        n = min(n_samples, total)
        indices = sample_indices(total, n)
    else:
        # Materialize only up to n_samples from the stream
        try:
            from itertools import islice
            data_buffer = list(islice(data, n_samples))
        except Exception:
            data_buffer = []
        total = len(data_buffer)
        n = total
        indices = list(range(n))
    outputs = []
    em_sum = 0
    f1_sum = 0.0
    rougeL_sum = 0.0
    inclusion_sum = 0
    latencies = []

    # process in batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        meta_list = []  # tuples (idx, answers, instruction)

        for idx in batch_idxs:
            ex = (data_buffer[int(idx)] if data_buffer is not None else data[int(idx)])
            instruction, answers, context, meta = extract_fields_for_dataset(ex, dataset_name)
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-question"})
                continue
            # MS MARCO-specific: prefer well-formed answers when available
            if ("microsoft/ms_marco" in dataset_name) or ("msmarco" in dataset_name):
                try:
                    wf = None
                    # Common field name is 'wellFormedAnswers' (list of strings). Handle a few variants defensively.
                    for k in ["wellFormedAnswers", "well_formed_answers", "wellformedanswers"]:
                        if isinstance(ex, dict) and k in ex and ex[k]:
                            wf = ex[k]
                            break
                    if wf:
                        if isinstance(wf, dict) and "text" in wf:
                            wf_list = wf["text"] if isinstance(wf["text"], list) else [wf["text"]]
                        elif isinstance(wf, list):
                            wf_list = [str(x) for x in wf if x is not None]
                        else:
                            wf_list = [str(wf)]
                        # Replace answers if well-formed are non-empty
                        if any(s and str(s).strip() for s in wf_list):
                            answers = wf_list
                    # Fallback to 'answers' if still empty; ensure list[str]
                    if not answers:
                        a = ex.get("answers") if isinstance(ex, dict) else None
                        if isinstance(a, dict):
                            if "text" in a:
                                answers = a["text"] if isinstance(a["text"], list) else [a["text"]]
                            elif "answer" in a:
                                answers = a["answer"] if isinstance(a["answer"], list) else [a["answer"]]
                        elif isinstance(a, list):
                            answers = [str(x) for x in a]
                        elif isinstance(a, str):
                            answers = [a]
                except Exception:
                    # Non-fatal; keep previously extracted answers
                    pass
            actual_prompts.append(format_prompt(instruction, paragraph=context))
            meta_list.append((idx, answers, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, answers, instruction), text in zip(meta_list, texts):
            if metric_mode == "em_f1":
                # MS MARCO: use strict EM (normalized equality only); F1 as usual
                if ("microsoft/ms_marco" in dataset_name) or ("msmarco" in dataset_name):
                    def _strict_best_em_f1(pred: str, golds: Optional[List[str]]):
                        if not golds:
                            return 0, 0.0
                        best_em = 0
                        best_f1 = 0.0
                        npred = normalize_answer(pred)
                        for g in golds:
                            ng = normalize_answer(g)
                            em_v = 1 if npred == ng else 0
                            f1_v = token_f1(pred, g)
                            if em_v > best_em:
                                best_em = em_v
                            if f1_v > best_f1:
                                best_f1 = f1_v
                        return best_em, best_f1
                    em, f1 = _strict_best_em_f1(text, answers)
                else:
                    em, f1 = best_em_f1_over_golds(text, answers)
                em_sum += em
                f1_sum += f1
                # Add ROUGE-L for MS MARCO (common scoring)
                if "microsoft/ms_marco" in dataset_name or "msmarco" in dataset_name:
                    rougeL = best_rouge_l_over_golds(text, answers)
                    rougeL_sum += rougeL
                else:
                    rougeL = None
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "em": em,
                    "f1": f1,
                    **({"rougeL": rougeL} if rougeL is not None else {}),
                    "latency_s": per_latency
                })
            elif metric_mode == "inclusion":
                inc = inclusion_match(text, answers)
                inclusion_sum += inc
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "inclusion": inc,
                    "latency_s": per_latency
                })
            else:
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "latency_s": per_latency
                })

    # Only count examples that have any non-empty gold answers for metric calculations
    def has_gold(o):
        g = o.get("gold")
        return isinstance(g, list) and any((isinstance(x, str) and x.strip()) for x in g)
    evaled = len([o for o in outputs if (not o.get("skipped", False)) and has_gold(o)])
    summary = {
        "dataset": dataset_name,
        "split": split,
        "examples_evaluated": evaled,
        "output_file": str(out_dir / f"{safe_filename(dataset_name)}_outputs.jsonl"),
    }
    if metric_mode == "em_f1":
        summary["em"] = (em_sum / evaled) if evaled else None
        summary["f1"] = (f1_sum / evaled) if evaled else None
        if evaled and ("microsoft/ms_marco" in dataset_name or "msmarco" in dataset_name):
            summary["rougeL"] = (rougeL_sum / evaled)
        summary["avg_latency_s"] = (sum(latencies) / len(latencies)) if latencies else None
    elif metric_mode == "inclusion":
        summary["inclusion_rate"] = (inclusion_sum / evaled) if evaled else None
        summary["avg_latency_s"] = (sum(latencies) / len(latencies)) if latencies else None

    out_file = out_dir / f"{safe_filename(dataset_name)}_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return summary

def evaluate_ms_marco(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int,
                      k_values=(1, 5, 10), ndcg_k_values=(5, 10)):
    """
    MS MARCO-style ranking evaluation (MRR@K, Recall@K, NDCG@K).

    Heuristics are used to find a ranked list of retrieved items and their relevance labels
    inside each dataset example. Common dataset field shapes supported (best-effort):
      - example["retrieved_documents"] -> list[{"is_relevant": 0/1} | {"score":..., "id":..., "relevance":...}]
      - example["passages"] or example["docs"] -> list[{"passage_text":..., "is_relevant": 0/1}]
      - example["positive_passages"] (list of positives) + example["passages"] (full ranked list)
      - example["qrels"] or dataset-level mapping is not automatically supported here

    Returned summary contains aggregated MRR@K, Recall@K and NDCG@K across evaluated examples.
    """
    import math

    def _try_extract_ranked_relevances(ex) -> Optional[List[float]]:
        """
        Return a list of relevance scores (higher == more relevant) in the order of the ranking.
        Prefer binary 0/1, but accept graded relevance if present.
        """
        # 1) Direct 'retrieved_documents' or 'retrieved_docs' or 'retrieved'
        for key in ("retrieved_documents", "retrieved_docs", "retrieved", "retrieval", "retrieved_items"):
            if key in ex and ex[key]:
                lst = ex[key]
                if isinstance(lst, (list, tuple)):
                    rels = []
                    for item in lst:
                        if isinstance(item, dict):
                            # common markers: 'is_relevant', 'relevance', 'label', 'score'
                            if "is_relevant" in item:
                                rels.append(int(bool(item.get("is_relevant"))))
                                continue
                            if "relevance" in item:
                                try:
                                    rels.append(float(item.get("relevance") or 0.0))
                                    continue
                                except Exception:
                                    pass
                            if "label" in item:
                                try:
                                    rels.append(int(item.get("label") or 0))
                                    continue
                                except Exception:
                                    pass
                            if "score" in item:
                                # score doesn't imply relevance but we keep it as proxy (higher is better)
                                try:
                                    rels.append(float(item.get("score") or 0.0))
                                    continue
                                except Exception:
                                    pass
                        # fallback: item could be a tuple (text, label) or [text, label]
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            # prefer second element if numeric-like
                            cand = item[1]
                            try:
                                rels.append(float(cand))
                                continue
                            except Exception:
                                # try boolean
                                try:
                                    rels.append(int(bool(cand)))
                                    continue
                                except Exception:
                                    pass
                    if rels:
                        return rels

        # 2) 'passages' / 'docs' lists where each passage may have 'is_relevant' or 'label'
        for key in ("passages", "docs", "passage_list", "documents", "candidates", "candidates_passages"):
            if key in ex and ex[key]:
                lst = ex[key]
                if isinstance(lst, (list, tuple)):
                    rels = []
                    for p in lst:
                        if isinstance(p, dict):
                            if "is_relevant" in p:
                                rels.append(int(bool(p.get("is_relevant"))))
                                continue
                            if "relevance" in p:
                                try:
                                    rels.append(float(p.get("relevance") or 0.0))
                                    continue
                                except Exception:
                                    pass
                            if "label" in p:
                                try:
                                    rels.append(int(p.get("label") or 0))
                                    continue
                                except Exception:
                                    pass
                            # sometimes binary flag stored as 'relevant'
                            if "relevant" in p:
                                rels.append(int(bool(p.get("relevant"))))
                                continue
                        # sometimes passage entry is a tuple [text, is_positive]
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            try:
                                rels.append(int(bool(p[1])))
                                continue
                            except Exception:
                                pass
                    if rels:
                        return rels

        # 3) MS MARCO format: 'positive_passages' contains positives (no ranking provided).
        # If dataset also provides 'passages' (ranked), we can mark relevance by membership in positives.
        if "positive_passages" in ex and "passages" in ex:
            positives = ex.get("positive_passages") or []
            passages = ex.get("passages") or []
            # create a quick set by normalized text or id
            pos_set = set()
            for p in positives:
                if isinstance(p, dict) and "passage_text" in p:
                    pos_set.add(normalize_answer(p["passage_text"]))
                elif isinstance(p, str):
                    pos_set.add(normalize_answer(p))
            rels = []
            for p in passages:
                if isinstance(p, dict) and "passage_text" in p:
                    rels.append(1 if normalize_answer(p["passage_text"]) in pos_set else 0)
                elif isinstance(p, str):
                    rels.append(1 if normalize_answer(p) in pos_set else 0)
            if rels:
                return rels

        # 4) As a last resort, try fields containing qid->relevance mapping (rare in HF in-example)
        # Not implemented: if your dump stores qrels separately, compute externally and pass mapping in.
        return None

    def _dcg_at_k(rels: List[float], k: int) -> float:
        dcg = 0.0
        for i in range(min(k, len(rels))):
            rel = rels[i]
            # graded relevance DCG
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
        return dcg

    def _idcg_at_k(rels: List[float], k: int) -> float:
        # ideal ordering is sorted descending
        ideal = sorted(rels, reverse=True)
        return _dcg_at_k(ideal, k)

    def _mrr_at_k_from_rels(rels: List[float], k: int) -> float:
        # find first position with relevance > 0
        for i in range(min(k, len(rels))):
            if rels[i] and float(rels[i]) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def _recall_at_k_from_rels(rels: List[float], k: int) -> float:
        # number of relevant items in top-k divided by total number of relevant items in whole list
        total_relevant = sum(1 for r in rels if float(r) > 0)
        if total_relevant == 0:
            return 0.0
        found = sum(1 for r in rels[:k] if float(r) > 0)
        return float(found) / float(total_relevant)

    print("Evaluating MS MARCO-style ranking metrics")
    split = choose_split(ds)
    data = ds[split]
    supports_index = hasattr(data, "__len__") and hasattr(data, "__getitem__")
    # materialize if needed
    if supports_index:
        total = len(data)
        n = min(n_samples, total)
        indices = sample_indices(total, n)
    else:
        # streaming dataset - convert first n_samples
        from itertools import islice
        data_buffer = list(islice(data, n_samples))
        total = len(data_buffer)
        indices = list(range(total))
        # support using `data_buffer` below
        data = {"_buffer": data_buffer}
        def _get_item(i):
            return data["_buffer"][i]
    outputs = []
    latencies = []

    # accumulators
    mrr_acc = {k: 0.0 for k in k_values}
    recall_acc = {k: 0.0 for k in k_values}
    ndcg_acc = {k: 0.0 for k in ndcg_k_values}
    evaluated = 0

    # iterate
    for idx in indices:
        if supports_index:
            ex = data[int(idx)]
        else:
            ex = _get_item(idx)

        # try to extract ranked relevance list
        rels = _try_extract_ranked_relevances(ex)
        if rels is None:
            # can't evaluate this example; record skip
            outputs.append({"index": idx, "skipped": True, "reason": "no-ranked-retrieval-found"})
            continue

        # compute metrics per example
        per_lat = None  # no model generation needed for ranking-only metrics in many datasets
        # if you want to evaluate a retriever that produced ranked results in the example,
        # we use those ranks directly.

        for k in k_values:
            mrr_acc[k] += _mrr_at_k_from_rels(rels, k)
            recall_acc[k] += _recall_at_k_from_rels(rels, k)

        for k in ndcg_k_values:
            idcg = _idcg_at_k(rels, k)
            dcg = _dcg_at_k(rels, k)
            ndcg_acc[k] += (dcg / idcg) if idcg > 0.0 else 0.0

        outputs.append({
            "index": idx,
            "ranked_relevance_list": rels,
            "latency_s": per_lat
        })
        evaluated += 1

    # aggregate
    if evaluated == 0:
        summary = {
            "dataset": "microsoft/ms_marco",
            "split": split,
            "examples_evaluated": 0,
            "note": "No evaluable examples found (no ranked retrieval fields detected).",
            "output_file": str(out_dir / f"{safe_filename('microsoft_ms_marco')}_outputs.jsonl")
        }
    else:
        summary = {
            "dataset": "microsoft/ms_marco",
            "split": split,
            "examples_evaluated": evaluated,
            "mrr": {k: (mrr_acc[k] / evaluated) for k in k_values},
            "recall": {k: (recall_acc[k] / evaluated) for k in k_values},
            "ndcg": {k: (ndcg_acc[k] / evaluated) for k in ndcg_k_values},
            "avg_latency_s": None,
            "output_file": str(out_dir / f"{safe_filename('microsoft_ms_marco')}_outputs.jsonl")
        }

    out_file = out_dir / f"{safe_filename('microsoft_ms_marco')}_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return summary

def evaluate_squad_v2(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """
    Evaluate SQuAD v2-style QA (handles unanswerable questions).

    - Uses the existing normalize_answer, token_f1, best_em_f1_over_golds utilities.
    - Unanswerable handling:
        * If the gold answers list is empty or example.marked 'is_impossible' (or similar),
          the example is considered unanswerable.
        * A predicted "no answer" counts as correct (EM=1, F1=1) for unanswerable examples.
          We treat a prediction as "no answer" if the normalized prediction is empty OR it
          contains explicit phrases like "no answer", "unanswerable", "cannot determine", "no_answer", "n/a".
    - Writes per-example JSONL to out_dir / "squad_v2_outputs.jsonl" and returns a summary dict.
    """
    print("Evaluating SQuAD v2 (EM & F1 with unanswerable handling)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    em_sum = 0
    f1_sum = 0.0
    latencies = []

    def pred_is_no_answer(pred_text: str) -> bool:
        npred = normalize_answer(pred_text)
        if not npred:
            return True
        # common explicit "no answer" signals
        if re.search(r"\b(no answer|no_answer|unanswerable|cannot determine|cannot be answered|n/?a|n\.a\.|n/a)\b", npred):
            return True
        return False

    # process in batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        prompts = []
        metas = []  # list of tuples (idx, ex, answers, is_unanswerable, instruction, context)

        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, "squad_v2")
            # SQuAD format uses 'question' and 'context' -> extract_fields_for_dataset should return them.
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-question"})
                continue

            # Determine unanswerable flag robustly
            is_unanswerable = False
            # explicit SQuAD v2 flag
            if isinstance(ex, dict):
                for k in ("is_impossible", "unanswerable", "is_unanswerable", "no_answer"):
                    if k in ex and ex[k] is not None:
                        v = ex[k]
                        if isinstance(v, bool):
                            is_unanswerable = bool(v)
                            break
                        try:
                            is_unanswerable = bool(int(v))
                            break
                        except Exception:
                            # strings like "true"/"false"
                            if isinstance(v, str):
                                vv = v.strip().lower()
                                if vv in ("true", "false"):
                                    is_unanswerable = (vv == "true")
                                    break
            # If answers is empty or None, treat as unanswerable
            if not answers:
                is_unanswerable = True

            prompts.append(format_prompt(instruction, paragraph=context))
            metas.append((idx, ex, answers, is_unanswerable, instruction))

        if not prompts:
            continue

        t0 = time.perf_counter()
        preds = batched_generate(model, prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(prompts))
        latencies.extend([per_latency] * len(prompts))

        for (idx, ex, answers, is_unanswerable, instruction), pred in zip(metas, preds):
            # determine prediction no-answer
            pred_no_ans = pred_is_no_answer(pred)

            if is_unanswerable:
                # gold is unanswerable
                if pred_no_ans:
                    em_v = 1
                    f1_v = 1.0
                else:
                    em_v = 0
                    f1_v = 0.0
            else:
                # answerable: compute best EM and F1 across gold answers
                # answers is expected to be a list[str]; guard against None
                golds = answers if isinstance(answers, list) else (answers or [])
                # If model predicted a no-answer when gold exists -> zero
                if pred_no_ans:
                    em_v = 0
                    f1_v = 0.0
                else:
                    em_v, f1_v = best_em_f1_over_golds(pred, golds)

            em_sum += em_v
            f1_sum += f1_v

            outputs.append({
                "index": idx,
                "instruction": instruction,
                "prediction": pred,
                "gold": answers,
                "is_unanswerable": is_unanswerable,
                "pred_is_no_answer": pred_no_ans,
                "em": em_v,
                "f1": f1_v,
                "latency_s": per_latency
            })

    # Count examples that were evaluated (not skipped)
    evaled = len([o for o in outputs if not o.get("skipped", False)])
    avg_em = (em_sum / evaled) if evaled else None
    avg_f1 = (f1_sum / evaled) if evaled else None

    out_file = out_dir / "squad_v2_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "squad_v2",
        "split": split,
        "examples_evaluated": evaled,
        "em": avg_em,
        "f1": avg_f1,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "output_file": str(out_file)
    }
    return summary

def evaluate_nq(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """
    Google Natural Questions (NQ) evaluation following the user's specification.

    Computes Short Answer (SA) and Long Answer (LA) EM and F1, handles nulls, and
    writes per-example outputs to `natural-questions_outputs.jsonl`.
    """
    print("Evaluating Natural Questions (short & long answers)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)

    outputs = []
    sa_em_sum = 0
    sa_f1_sum = 0.0
    la_em_sum = 0
    la_f1_sum = 0.0
    sa_answerable_correct = 0
    la_answerable_correct = 0
    latencies = []

    # Use the canonical normalizer and token_f1 defined above
    def exact_match(pred: str, gold: str) -> int:
        return 1 if normalize_answer(pred) == normalize_answer(gold) else 0

    def max_metric_over_answers(pred: str, golds: Optional[List[str]], metric_fn):
        if not golds:
            # For EM and F1, if there are no golds, treat as 0 (except when both pred and gold are null handled elsewhere)
            return 0 if metric_fn is exact_match else 0.0
        scores = [metric_fn(pred, g) for g in golds]
        return max(scores)

    # process in batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        prompts = []
        metas = []  # (idx, ex, gold_short_list, gold_long_list)

        for idx in batch_idxs:
            ex = data[int(idx)]
            # Try to extract short and long answers from common fields
            gold_short = None
            gold_long = None
            # HF copies of NQ often store 'short_answers' and 'long_answer'
            if isinstance(ex, dict):
                # short answers: list of dicts with 'text' or span strings
                if 'short_answers' in ex and ex['short_answers']:
                    sa = ex['short_answers']
                    if isinstance(sa, list):
                        gold_short = []
                        for it in sa:
                            if isinstance(it, dict) and 'text' in it:
                                gold_short.append(str(it['text']))
                            elif isinstance(it, str):
                                gold_short.append(it)
                # long answer may be stored as a dict with 'text' or 'candidate' info
                if 'long_answer' in ex and ex['long_answer']:
                    la = ex['long_answer']
                    if isinstance(la, dict) and 'text' in la:
                        gold_long = [str(la['text'])]
                    elif isinstance(la, str):
                        gold_long = [la]
                # fallback keys
                if not gold_short and 'annotations' in ex:
                    ann = ex['annotations']
                    if isinstance(ann, list) and ann:
                        # collect short/long from first annotation as fallback
                        a0 = ann[0]
                        if isinstance(a0, dict):
                            if 'short_answers' in a0 and a0['short_answers']:
                                gold_short = [str(x.get('text')) for x in a0.get('short_answers') if x and x.get('text')]
                            if 'long_answer' in a0 and a0['long_answer'] and isinstance(a0['long_answer'], dict) and a0['long_answer'].get('text'):
                                gold_long = [str(a0['long_answer']['text'])]

            # normalize to lists
            if gold_short is None:
                gold_short = []
            if gold_long is None:
                gold_long = []

            # Build a prompt: ask model to return structured JSON with token-span offsets
            # We explicitly request token offsets using whitespace tokenization of the paragraph
            # so outputs can be exported directly to the official NQ prediction format.
            paragraph_text = ex.get('document') or ex.get('context') or ex.get('paragraph') or ex.get('document_text') or ex.get('long_answer_context') or ''
            # Truncate long paragraphs to avoid exceeding model context length (vLLM default ~4096 tokens).
            # Keep the last portion of the paragraph (likely contains answers) and explicitly tell the model
            # that we truncated the paragraph to ensure token offsets refer to the included text.
            def _truncate_paragraph(p: str, max_chars: int = 2000) -> str:
                if not p:
                    return ''
                p = str(p)
                if len(p) <= max_chars:
                    return p
                # keep the last max_chars characters with a small prefix marker
                return p[-max_chars:]

            paragraph_included = _truncate_paragraph(paragraph_text, max_chars=2000)
            trunc_note = '' if paragraph_included == paragraph_text else '(NOTE: paragraph truncated to last 2000 chars)\n'
            # We ask for character offsets (start_char/end_char) which are easier for the model
            # to produce reliably; we'll convert them to whitespace-token indices locally.
            prompt = format_prompt(
                f"Given the paragraph below, answer the question and return a JSON object with the following keys:\n"
                f"  \"example_id\": integer id for this example,\n"
                f"  \"long_answer\": {{\"start_char\": int, \"end_char\": int}},\n"
                f"  \"long_answer_score\": float,\n"
                f"  \"short_answers\": [{{\"start_char\": int, \"end_char\": int}},...],\n"
                f"  \"short_answers_score\": float,\n"
                f"  \"yes_no_answer\": \"NONE\" or \"YES\" or \"NO\"\n"
                f"IMPORTANT: Character offsets must be character indices into the paragraph exactly as shown below (0-based).\n"
                f"Return only the JSON object and nothing else.\n\n{trunc_note}Paragraph:\n{paragraph_included}\n\nQuestion: {ex.get('question') if isinstance(ex, dict) else str(ex)}"
            )
            prompts.append(prompt)
            metas.append((idx, ex, gold_short, gold_long))

        if not prompts:
            continue

        t0 = time.perf_counter()
        preds = batched_generate(model, prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(prompts))
        latencies.extend([per_latency] * len(prompts))

        for (idx, ex, gold_short, gold_long), pred in zip(metas, preds):
            # Try to parse JSON first (preferred). If JSON parsing fails, strip noisy
            # retrieval/evidence tags and fall back to heuristic regex extraction.
            sa_pred = ''
            la_pred = ''
            try:
                parsed = json.loads(pred.strip())
                # Respect both 'short_answers' and 'short_answer' variants
                if isinstance(parsed, dict):
                    # normalize and validate example_id
                    example_id = None
                    for k in ('example_id', 'id', 'exampleId'):
                        if k in parsed:
                            example_id = parsed[k]
                            break
                    # fallback to dataset-provided ids or index
                    if example_id is None:
                        # try common dataset keys
                        example_id = ex.get('example_id') or ex.get('id') or ex.get('exampleId') or idx
                    parsed['example_id'] = example_id
                    if 'short_answers' in parsed and parsed['short_answers']:
                        # ensure list[str]
                        sa_list = parsed['short_answers']
                        if isinstance(sa_list, list):
                            sa_pred = ' '.join([str(x) for x in sa_list if x])
                        else:
                            sa_pred = str(sa_list)
                    elif 'short_answer' in parsed:
                        sa_pred = str(parsed.get('short_answer') or '')
                    la_pred = str(parsed.get('long_answer') or parsed.get('long_answer_text') or '')
                    # store parsed JSON for export
                    nq_pred_item = parsed
                else:
                    # not a dict: fall through to heuristics
                    raise ValueError('parsed JSON not dict')
            except Exception:
                # fallback: remove common noisy markers like [Retrieval]<paragraph> and bracketed tags
                cleaned = re.sub(r"\[Retrieval\].*?\</paragraph\>", "", pred, flags=re.I | re.S)
                cleaned = re.sub(r"\[Retrieval\]<paragraph>.*?</paragraph>", "", cleaned, flags=re.I | re.S)
                cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
                cleaned = re.sub(r"\\s+", " ", cleaned).strip()
                # Heuristic regex as before
                m_sa = re.search(r"SHORT_ANSWER[:\-\s]*\[?(.*?)\]?\s*(?:LONG_ANSWER:|$)", cleaned, flags=re.I | re.S)
                if m_sa:
                    sa_pred = m_sa.group(1).strip()
                else:
                    sa_pred = cleaned.split('\n')[0].strip()
                m_la = re.search(r"LONG_ANSWER[:\-\s]*(.*)$", cleaned, flags=re.I | re.S)
                if m_la:
                    la_pred = m_la.group(1).strip()
                else:
                    parts = cleaned.split('\n', 1)
                    la_pred = parts[1].strip() if len(parts) > 1 else ''
                    nq_pred_item = None

            # Determine null (unanswerable) for gold: treat empty gold_short & empty gold_long as null
            gold_is_null = (not gold_short) and (not gold_long)
            pred_sa_is_null = not sa_pred or normalize_answer(sa_pred) == ''
            pred_la_is_null = not la_pred or normalize_answer(la_pred) == ''

            # Short answer metrics: EM & F1 (max over golds)
            if gold_is_null:
                sa_em = 1 if pred_sa_is_null else 0
                sa_f1 = 1.0 if pred_sa_is_null else 0.0
            else:
                # if gold_short exists, match against them; otherwise consider gold_long as reference
                if gold_short:
                    # EM: exact match to any gold short
                    sa_em = max_metric_over_answers(sa_pred, gold_short, exact_match)
                    # F1: token-level F1, reuse token_f1 utility
                    sa_f1 = max([token_f1(sa_pred, g) for g in gold_short]) if gold_short else 0.0
                else:
                    if gold_long:
                        sa_em = max_metric_over_answers(sa_pred, gold_long, exact_match)
                        sa_f1 = max([token_f1(sa_pred, g) for g in gold_long]) if gold_long else 0.0
                    else:
                        sa_em = 0
                        sa_f1 = 0.0

            # Long answer metrics
            if gold_is_null:
                la_em = 1 if pred_la_is_null else 0
                la_f1 = 1.0 if pred_la_is_null else 0.0
            else:
                if gold_long:
                    la_em = max_metric_over_answers(la_pred, gold_long, exact_match)
                    la_f1 = max([token_f1(la_pred, g) for g in gold_long]) if gold_long else 0.0
                else:
                    if gold_short:
                        concat_short = ' '.join(gold_short)
                        la_em = exact_match(la_pred, concat_short)
                        la_f1 = token_f1(la_pred, concat_short)
                    else:
                        la_em = 0
                        la_f1 = 0.0

            sa_em_sum += sa_em
            sa_f1_sum += sa_f1
            la_em_sum += la_em
            la_f1_sum += la_f1

            outputs.append({
                "index": idx,
                "question": ex.get('question') if isinstance(ex, dict) else str(ex),
                "prediction_raw": pred,
                "pred_short": sa_pred,
                "pred_long": la_pred,
                "gold_short": gold_short,
                "gold_long": gold_long,
                "short_em": sa_em,
                "short_f1": sa_f1,
                "long_em": la_em,
                "long_f1": la_f1,
                "latency_s": per_latency
            })
            # collect official-format predictions if the model returned JSON spans
            if 'nq_pred_item' in locals() and nq_pred_item:
                # Normalize to official field names (start_token/end_token ints). Accept char spans too.
                def _safe_int(x):
                    try:
                        return int(x)
                    except Exception:
                        return -1

                pred_entry = {"example_id": int(nq_pred_item.get('example_id', idx))}
                # helper: map char offset to whitespace-token index in the included paragraph
                def _char_to_token_index(paragraph: str, char_pos: int) -> int:
                    if paragraph is None or char_pos is None:
                        return -1
                    try:
                        import re
                        # build token spans for whitespace-tokenization
                        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", paragraph)]
                        for ti, (sstart, send) in enumerate(spans):
                            if sstart <= char_pos < send:
                                return ti
                        # if char_pos at or after last token end, return last token index
                        if spans and char_pos >= spans[-1][1]:
                            return len(spans) - 1
                        return -1
                    except Exception:
                        return -1

                # long_answer may provide char spans or token spans
                la = nq_pred_item.get('long_answer')
                if isinstance(la, dict):
                    if 'start_token' in la or 'end_token' in la:
                        pred_entry['long_answer'] = {"start_token": _safe_int(la.get('start_token', -1)), "end_token": _safe_int(la.get('end_token', -1))}
                    elif 'start_char' in la or 'end_char' in la:
                        sc = la.get('start_char', -1)
                        ec = la.get('end_char', -1)
                        start_t = _char_to_token_index(paragraph_included, _safe_int(sc))
                        end_t = _char_to_token_index(paragraph_included, _safe_int(ec))
                        pred_entry['long_answer'] = {"start_token": start_t, "end_token": end_t}
                    else:
                        pred_entry['long_answer'] = {"start_token": -1, "end_token": -1}
                else:
                    pred_entry['long_answer'] = {"start_token": -1, "end_token": -1}
                pred_entry['long_answer_score'] = float(nq_pred_item.get('long_answer_score') or 0.0)
                # short answers: accept list of dicts with char spans or token spans
                sa_list = nq_pred_item.get('short_answers') or nq_pred_item.get('short_answer') or []
                out_sa = []
                if isinstance(sa_list, list):
                    for it in sa_list:
                        if isinstance(it, dict):
                            if 'start_token' in it or 'end_token' in it:
                                out_sa.append({"start_token": _safe_int(it.get('start_token', -1)), "end_token": _safe_int(it.get('end_token', -1))})
                            elif 'start_char' in it or 'end_char' in it:
                                sc = it.get('start_char', -1)
                                ec = it.get('end_char', -1)
                                out_sa.append({"start_token": _char_to_token_index(paragraph_included, _safe_int(sc)), "end_token": _char_to_token_index(paragraph_included, _safe_int(ec))})
                pred_entry['short_answers'] = out_sa
                pred_entry['short_answers_score'] = float(nq_pred_item.get('short_answers_score') or nq_pred_item.get('short_answers_score', 0.0) or 0.0)
                pred_entry['yes_no_answer'] = str(nq_pred_item.get('yes_no_answer') or 'NONE')
                # collect in a list in outer scope
                if 'nq_official_predictions' not in locals():
                    nq_official_predictions = []
                nq_official_predictions.append(pred_entry)

    evaled = len([o for o in outputs if not o.get('skipped', False)])
    sa_em_avg = (sa_em_sum / evaled) if evaled else None
    sa_f1_avg = (sa_f1_sum / evaled) if evaled else None
    la_em_avg = (la_em_sum / evaled) if evaled else None
    la_f1_avg = (la_f1_sum / evaled) if evaled else None

    # Combined metrics (average of short and long)
    combined_em = None
    combined_f1 = None
    if sa_em_avg is not None and la_em_avg is not None:
        combined_em = (sa_em_avg + la_em_avg) / 2.0
    if sa_f1_avg is not None and la_f1_avg is not None:
        combined_f1 = (sa_f1_avg + la_f1_avg) / 2.0

    out_file = out_dir / "natural-questions_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    # If the model returned official-format NQ prediction items (with token/byte spans),
    # write them to a separate JSON file suitable for the official NQ evaluator.
    official_preds_file = out_dir / "nq_official_predictions.json"
    if 'nq_official_predictions' in locals() and nq_official_predictions:
        # official evaluator expects a top-level dict with key 'predictions' mapping to a list
        with open(official_preds_file, 'w', encoding='utf-8') as fh:
            json.dump({'predictions': nq_official_predictions}, fh, ensure_ascii=False, indent=2)
        official_preds_path = str(official_preds_file)
    else:
        official_preds_path = None

    return {
        "dataset": "google-research-datasets/natural_questions",
        "split": split,
        "examples_evaluated": evaled,
        "short_em": sa_em_avg,
        "short_f1": sa_f1_avg,
        "long_em": la_em_avg,
        "long_f1": la_f1_avg,
        "combined_em": combined_em,
        "combined_f1": combined_f1,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "output_file": str(out_file),
        "official_predictions_file": official_preds_path,
    }

def evaluate_race(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """
    RACE multiple-choice evaluation.
    Expects model predictions as integer indices 0-3 for each example.
    This evaluator will prompt the model with the passage+question+options and parse a predicted index.
    Returns accuracy, macro precision/recall/f1 and confusion matrix.
    """
    print("Evaluating RACE (multiple-choice)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)

    preds = []
    golds = []
    outputs = []
    latencies = []

    # simple parser: look for digit 0-3 or option letter A-D
    def parse_pred(text: str) -> Optional[int]:
        if text is None:
            return None
        t = text.strip()
        # try find single digit 0-3
        m = re.search(r"\b([0-3])\b", t)
        if m:
            return int(m.group(1))
        # try letter A-D
        m2 = re.search(r"\b([A-Da-d])\b", t)
        if m2:
            c = m2.group(1).upper()
            return ord(c) - ord('A')
        # try words 'option 1' etc
        m3 = re.search(r"option\s*([1-4])", t, flags=re.I)
        if m3:
            return int(m3.group(1)) - 1
        return None

    # batching
    for i in range(0, len(indices), batch_size):
        batch = indices[i:i+batch_size]
        prompts = []
        metas = []
        for idx in batch:
            ex = data[int(idx)]
            passage = ex.get('article') or ex.get('passage') or ex.get('context') or ''
            question = ex.get('question') or ex.get('query') or ''
            options = ex.get('options') or ex.get('choices') or []
            if not options:
                # fallback: try fields 'options' with keys
                opts = []
                for k in ['option1','option2','option3','option4']:
                    if k in ex:
                        opts.append(ex[k])
                options = opts
            # build prompt
            opt_text = '\n'.join([f"{i}. {o}" for i,o in enumerate(options)])
            prompt = format_prompt(f"Passage:\n{passage}\n\nQuestion: {question}\nOptions:\n{opt_text}\nAnswer with the option index 0-3.")
            prompts.append(prompt)
            metas.append((idx, ex, options))

        t0 = time.perf_counter()
        texts = batched_generate(model, prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(prompts))
        latencies.extend([per_latency]*len(prompts))

        for (idx, ex, options), text in zip(metas, texts):
            pred_idx = parse_pred(text)
            gold = ex.get('answer')
            try:
                gold_idx = int(gold)
            except Exception:
                # sometimes deposited as 'A','B','C','D'
                if isinstance(gold, str) and gold.upper() in ('A','B','C','D'):
                    gold_idx = ord(gold.upper()) - ord('A')
                else:
                    gold_idx = None
            preds.append(pred_idx if pred_idx is not None else -1)
            golds.append(gold_idx if gold_idx is not None else -1)
            outputs.append({
                'index': idx,
                'question': ex.get('question'),
                'options': options,
                'gold': gold_idx,
                'pred': pred_idx,
                'raw': text,
                'latency_s': per_latency
            })

    # compute metrics, filter out examples where gold is invalid (-1)
    paired = [(g,p) for g,p in zip(golds,preds) if g is not None and g >= 0]
    if not paired:
        summary = {
            'dataset': 'ehovy/race',
            'split': split,
            'examples_evaluated': 0,
            'note': 'No evaluable examples found',
            'output_file': str(out_dir / 'race_outputs.jsonl')
        }
        return summary

    golds_f = [g for g,p in paired]
    preds_f = [p for g,p in paired]

    # accuracy
    correct = sum(1 for g,p in zip(golds_f,preds_f) if g == p)
    accuracy = correct / len(golds_f)

    # confusion matrix 4x4
    import math
    K = 4
    conf = [[0]*K for _ in range(K)]
    for g,p in zip(golds_f,preds_f):
        if 0 <= g < K and 0 <= p < K:
            conf[g][p] += 1

    # per-class precision/recall/f1 (macro)
    precisions = []
    recalls = []
    f1s = []
    for c in range(K):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(K) if r != c)
        fn = sum(conf[c][r] for r in range(K) if r != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precision_macro = sum(precisions) / K
    recall_macro = sum(recalls) / K
    f1_macro = sum(f1s) / K

    out_file = out_dir / 'race_outputs.jsonl'
    with open(out_file, 'w', encoding='utf-8') as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        'dataset': 'ehovy/race',
        'split': split,
        'examples_evaluated': len(golds_f),
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': conf,
        'avg_latency_s': (sum(latencies)/len(latencies)) if latencies else None,
        'output_file': str(out_file)
    }

def _parse_ragtruth_gold_label(ex: dict) -> Optional[int]:
    """Parse RAGTruth gold: 1 if any hallucination labels present, else 0. Returns None if unknown."""
    if not isinstance(ex, dict):
        return None
    # Candidates: 'hallucination_labels', 'hallucination_labels_processed'
    for k in ["hallucination_labels", "hallucination_labels_processed"]:
        if k in ex and ex[k] is not None:
            v = ex[k]
            try:
                # Some fields are JSON strings
                if isinstance(v, str):
                    v_parsed = json.loads(v)
                else:
                    v_parsed = v
                if isinstance(v_parsed, list):
                    return 1 if len(v_parsed) > 0 else 0
                # If dict with spans, consider any key presence
                if isinstance(v_parsed, dict):
                    return 1 if len(v_parsed) > 0 else 0
            except Exception:
                # If parsing fails, treat non-empty string as positive signal
                if isinstance(v, str) and v.strip():
                    # Heuristic: if it looks like a JSON list and contains '{' assume positive
                    return 1 if "{" in v or "[" in v else 0
    # Additional common single-label fields used in processed variants
    # Try boolean/numeric or string forms indicating hallucination presence
    for k in ["is_hallucinated", "hallucinated", "has_hallucination", "response_label", "response_label_processed", "label", "gold_label"]:
        if k in ex and ex[k] is not None:
            val = ex[k]
            # direct boolean/numeric
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return int(bool(val))
            # strings like "hallucinated" / "clean" / "no_hallucination" etc.
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("hallucinated", "hallucination", "has_hallucination", "yes", "true", "1", "positive", "pos", "halluc"):
                    return 1
                if v in ("not hallucinated", "no_hallucination", "clean", "no", "false", "0", "negative", "neg", "non-hallucinated"):
                    return 0
    return None

def evaluate_ragtruth(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """
    RAGTruth: compute response-level detection P/R/F1, plus try span-level if annotations exist.
    We'll look for a label field indicating hallucinated vs not.
    """
    print("Evaluating RAGTruth (response-level + optional span-level)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    tp = fp = fn = 0
    latencies = []
    span_prf_accum: List[Tuple[float, float, float]] = []

    # process batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        meta_list = []  # (idx, ex, meta, instruction)

        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, "wandb/RAGTruth-processed")
            if instruction is None and "query" in ex:
                instruction = ex.get("query")
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-query"})
                continue
            actual_prompts.append(format_prompt(instruction, paragraph=context))
            meta_list.append((idx, ex, meta, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, ex, meta, instruction), text in zip(meta_list, texts):
            # robust gold parse from dataset-specific fields, with fallback to detected meta label
            gold_binary: Optional[int] = _parse_ragtruth_gold_label(ex)
            if gold_binary is None and isinstance(meta, dict):
                # use previously detected heuristic label if available
                maybe_meta = meta.get("ragtruth_label")
                if maybe_meta is not None:
                    try:
                        gold_binary = int(bool(maybe_meta))
                    except Exception:
                        gold_binary = None
            gold_label = gold_binary  # store for output readability
            # naive predicted detection heuristic:
            lower_txt = text.lower()
            if any(phrase in lower_txt for phrase in ["i don't know", "i do not know", "cannot determine", "no retrieval", "no evidence", "i'm not sure", "not enough information", "i cannot find"]):
                pred_halluc = 0
            else:
                pred_halluc = 1
            # update confusion
            if gold_binary is not None:
                if pred_halluc == 1 and gold_binary == 1:
                    tp += 1
                elif pred_halluc == 1 and gold_binary == 0:
                    fp += 1
                elif pred_halluc == 0 and gold_binary == 1:
                    fn += 1
            outputs.append({
                "index": idx,
                "instruction": instruction,
                "prediction": text,
                "gold_label": gold_label,
                "pred_halluc": pred_halluc,
                "latency_s": per_latency,
                "ragtruth_spans": (meta.get("ragtruth_spans") if isinstance(meta, dict) else None)
            })
            if isinstance(meta, dict) and meta.get("ragtruth_spans"):
                prf = compute_span_level_prf(text, meta.get("ragtruth_spans"))
                if prf is not None:
                    span_prf_accum.append(prf)

    evaled = len([o for o in outputs if not o.get("skipped", False)])
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = None

    span_prf_avg = None
    span_prf_computed = False
    if span_prf_accum:
        ps = [p for p, r, f in span_prf_accum]
        rs = [r for p, r, f in span_prf_accum]
        fs = [f for p, r, f in span_prf_accum]
        span_prf_avg = {"p": sum(ps) / len(ps), "r": sum(rs) / len(rs), "f1": sum(fs) / len(fs)}
        span_prf_computed = True

    out_file = out_dir / "ragtruth_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        "dataset": "wandb/RAGTruth-processed",
        "split": split,
        "examples_evaluated": evaled,
        "response_precision": precision,
        "response_recall": recall,
        "response_f1": f1,
        "span_prf_avg": span_prf_avg,
        "span_prf_computed": span_prf_computed,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "output_file": str(out_file)
    }

# ---------------- Helpers ----------------
def sample_indices(total: int, n: int, seed: int = 42) -> List[int]:
    random.seed(seed)
    idxs = list(range(total))
    random.shuffle(idxs)
    return idxs[:n]

def format_prompt(instruction: str, paragraph: Optional[str] = None) -> str:
    # Follows the Self-RAG README prompt format.
    p = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    if paragraph:
        p += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return p

def choose_split(raw_ds) -> str:
    """Pick a split, preferring labeled splits over test.

    Many HF datasets hide labels in the test split. To avoid computing
    misleading zeros, we de-prioritize 'test' and prefer validation/dev/train.
    """
    for s in ["validation", "dev", "train", "validation_matched", "validation_unmatched", "test"]:
        if s in raw_ds:
            return s
    # default to first available
    return list(raw_ds.keys())[0]

def safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from dataset name."""
    return re.sub(r"[^\w\-_\.]", "_", name)

def coerce_dtype_for_vllm(dtype_arg: str) -> str:
    """
    Accept README-style dtype aliases while ensuring compatibility with vLLM.
    - "half" or "fp16" -> "float16"
    - "bf16" -> "bfloat16"
    - pass-through: "auto", "float16", "bfloat16", "float32"
    """
    x = (dtype_arg or "").strip().lower()
    if x in ("half", "fp16"):
        return "float16"
    if x in ("bf16",):
        return "bfloat16"
    if x in ("auto", "float16", "bfloat16", "float32"):
        return x
    # Fall back to float16 with a warning
    print(f"Warning: dtype '{dtype_arg}' not recognized. Falling back to float16.", file=sys.stderr)
    return "float16"

def setup_hf_auth(token: Optional[str]) -> None:
    """Configure Hugging Face auth from a provided token or environment.

    - If a token is provided (arg or env), export it to common env vars and
      attempt to persist/login via huggingface_hub when available.
    - If no token is found, we proceed assuming public access or prior CLI login.
    """
    try:
        tok = token or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if tok:
            # Export to both common env var names so downstream libraries pick it up
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
            os.environ["HF_TOKEN"] = tok
            # Persist/login if possible
            try:
                if HfFolder is not None:
                    HfFolder.save_token(tok)
                if hf_login is not None:
                    # Avoid touching git credential store
                    hf_login(token=tok, add_to_git_credential=False)
                print("Hugging Face token configured from argument/env.")
            except Exception as ie:
                print(f"Warning: could not persist/login Hugging Face token: {ie}", file=sys.stderr)
        else:
            print("No HF token provided; relying on prior `huggingface-cli login` or public access.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: setup_hf_auth encountered an issue: {e}", file=sys.stderr)

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Research-grade SELF-RAG evaluation (inference only).")
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b", help="HuggingFace model name for SELF-RAG")
    parser.add_argument("--download_dir", type=str, default=None, help="Optional model download cache directory")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples per dataset (max)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for generation")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="dtype for vllm (README default 'half'; accepted: half/fp16, bf16, auto, float16, bfloat16, float32)")
    parser.add_argument("--output_dir", type=str, default="./selfrag_eval_outputs", help="Directory to save outputs and summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HUGGING_FACE_HUB_TOKEN / HF_TOKEN)")
    parser.add_argument("--run_official_eval", action="store_true", help="If set, run official evaluators for datasets when gold files are provided")
    parser.add_argument("--trivia_gold", type=str, default=None, help="Path to official TriviaQA dataset JSON for evaluation")
    parser.add_argument("--fever_gold", type=str, default=None, help="Path to FEVER gold dev.jsonl for official FEVER scoring")
    parser.add_argument("--hotpot_gold_distractor", type=str, default=None, help="Path to HotPotQA distractor gold JSON")
    parser.add_argument("--hotpot_gold_fullwiki", type=str, default=None, help="Path to HotPotQA fullwiki gold JSON")
    parser.add_argument("--nq_gold", type=str, default=None, help="Path to Natural Questions gold file(s) (glob or dir as expected by official script)")
    parser.add_argument("--ragtruth_model", type=str, default=None, help="Local model path used for RAGTruth predictor (e.g. baseline)")
    parser.add_argument("--ragtruth_tokenizer", type=str, default="meta-llama/Llama-2-13b-hf", help="Tokenizer to use with predict_and_evaluate.py")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure HF auth (for private/authorized models or gated datasets)
    setup_hf_auth(args.hf_token)

    # instantiate vllm model
    llm_kwargs: Dict[str, Any] = {}
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    coerced_dtype = coerce_dtype_for_vllm(args.dtype)
    if coerced_dtype != (args.dtype or "").strip().lower():
        print(f"Coercing dtype '{args.dtype}' to '{coerced_dtype}' for vLLM compatibility.")

    print(f"Loading model {args.model_name} with dtype={args.dtype} (vLLM uses: {coerced_dtype}) ...")
    model = LLM(args.model_name, dtype=coerced_dtype, **llm_kwargs)
    # Follow README: skip_special_tokens=False to preserve Self-RAG reflection tokens in outputs.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, skip_special_tokens=False)

    # Datasets list (keep names the same)
    datasets_to_run = [
        ("mwong/fever-evidence-related", "fever"),
    ("microsoft/ms_marco", "msmarco_v2.1"),
    ("squad_v2", "squad_v2"),
    ("ehovy/race", "race"),
    ("hotpotqa/hotpot_qa", "hotpot_distractor_and_fullwiki"),
        ("wandb/RAGTruth-processed", "ragtruth"),
        ("mandarjoshi/trivia_qa", "trivia_rc"),
        ("google-research-datasets/natural_questions", "natural_questions")
    ]

    summary_list: List[Dict[str, Any]] = []

    for ds_id, tag in datasets_to_run:
        try:
            if ds_id == "ehovy/race":
                print("\n--- Loading RACE (config: all)\n")
                raw = load_dataset(ds_id, "all")
                summ = evaluate_race(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue
            if ds_id == "hotpotqa/hotpot_qa":
                # generation-only for HotPotQA; official scoring will be delegated to the official script.
                for cfg in ["distractor", "fullwiki"]:
                    print(f"\n--- Loading HotPotQA config: {cfg} (generation only, no internal scoring)")
                    raw = load_dataset(ds_id, cfg)
                    # Use metric_mode that does not compute internal metrics; this writes the outputs JSONL only.
                    summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=f"{ds_id}__{cfg}", metric_mode="none")
                    if summ:
                        summary_list.append(summ)
                continue

            if ds_id == "microsoft/ms_marco":
                print("\n--- Loading MS MARCO v2.1")
                # Load MS MARCO explicitly as requested
                ds = load_dataset("microsoft/ms_marco", "v2.1")
                summ = evaluate_ms_marco(ds, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "squad_v2":
                print("\n--- Loading SQuAD v2")
                raw = load_dataset("squad_v2")
                summ = evaluate_squad_v2(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "mwong/fever-evidence-related":
                print("\n--- Loading FEVER")
                raw = load_dataset(ds_id)
                summ = evaluate_fever(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "wandb/RAGTruth-processed":
                print("\n--- Loading RAGTruth")
                raw = load_dataset(ds_id)
                summ = evaluate_ragtruth(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "mandarjoshi/trivia_qa":
                print("\n--- Loading TriviaQA (rc) (generation only, no internal scoring)")
                raw = {"validation": load_dataset(ds_id, "rc", split="validation", streaming=True)}
                summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=f"{ds_id}::rc", metric_mode="none")
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "google-research-datasets/natural_questions":
                print("\n--- Loading Natural Questions (official)\n--- using config: default (generation + official-predictions export)")
                raw = load_dataset(ds_id, "default")
                summ = evaluate_nq(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                # For Natural Questions we keep generation but avoid adding internal aggregated metrics to the main summary
                if summ:
                    # Only keep pointer to official predictions file (if created) and basic bookkeeping in the summary
                    minimal = {
                        'dataset': summ.get('dataset'),
                        'split': summ.get('split'),
                        'examples_evaluated': summ.get('examples_evaluated'),
                        'output_file': summ.get('output_file'),
                        'official_predictions_file': summ.get('official_predictions_file'),
                        'avg_latency_s': summ.get('avg_latency_s')
                    }
                    summary_list.append(minimal)
                continue

            # default fallback
            raw = load_dataset(ds_id)
            summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=ds_id, metric_mode="em_f1")
            if summ:
                summary_list.append(summ)
        except Exception as e:
            print(f"Failed to evaluate {ds_id}: {e}", file=sys.stderr)

    # Save overall summary
    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_list, fh, indent=2)

    # Save CSV summary (flatten)
    summary_csv = out_dir / "summary.csv"
    # collect union of possible keys
    keys = set()
    for s in summary_list:
        keys.update(s.keys())
    keys = sorted(list(keys))
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for s in summary_list:
            writer.writerow(s)

    print("Evaluation complete. Outputs and summary saved to:", out_dir.resolve())
    # --- Official evaluator invocation (optional) ---
    if args.run_official_eval:
        print("Running official evaluators (where gold files provided)...")
        official_out_dir = out_dir / "official_eval"
        official_out_dir.mkdir(parents=True, exist_ok=True)
        # TriviaQA
        try:
            if args.trivia_gold:
                trivia_ds_name = "mandarjoshi/trivia_qa::rc"
                trivia_jsonl = out_dir / f"{safe_filename(trivia_ds_name)}_outputs.jsonl"
                if trivia_jsonl.exists():
                    cmd = [sys.executable, str(Path(__file__).parent / 'tools' / 'run_trivia_eval.py'), '--jsonl', str(trivia_jsonl), '--dataset', args.trivia_gold, '--out_dir', str(official_out_dir / 'trivia')]
                    print('Running TriviaQA official eval...')
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    open(official_out_dir / 'trivia_stdout.txt', 'w', encoding='utf-8').write(p.stdout + '\n' + p.stderr)
                    summary_list.append({
                        'dataset': trivia_ds_name,
                        'official_eval_stdout_file': str(official_out_dir / 'trivia_stdout.txt')
                    })
                else:
                    print(f"TriviaQA JSONL not found: {trivia_jsonl}")
            else:
                print('No TriviaQA gold path provided; skipping official TriviaQA eval.')
        except Exception as e:
            print('TriviaQA official eval failed:', e)

        # FEVER official scorer
        try:
            if args.fever_gold:
                fever_ds_name = "mwong/fever-evidence-related"
                fever_jsonl = out_dir / f"{safe_filename(fever_ds_name)}_outputs.jsonl"
                if fever_jsonl.exists():
                    cmd = [sys.executable, str(Path(__file__).parent / 'tools' / 'run_fever_eval.py'), '--jsonl', str(fever_jsonl), '--gold', args.fever_gold, '--out_dir', str(official_out_dir / 'fever')]
                    print('Running FEVER official scorer...')
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    open(official_out_dir / 'fever_stdout.txt', 'w', encoding='utf-8').write(p.stdout + '\n' + p.stderr)
                    # try to read fever_metrics.json produced by the runner
                    fever_metrics_path = official_out_dir / 'fever' / 'fever_metrics.json'
                    fever_metrics = None
                    if fever_metrics_path.exists():
                        try:
                            fever_metrics = json.loads(fever_metrics_path.read_text(encoding='utf-8'))
                        except Exception:
                            fever_metrics = None
                    entry = {'dataset': fever_ds_name, 'official_eval_stdout_file': str(official_out_dir / 'fever_stdout.txt')}
                    if fever_metrics is not None:
                        entry['official_metrics'] = fever_metrics
                    else:
                        entry['official_metrics_json'] = str(fever_metrics_path)
                    summary_list.append(entry)
                else:
                    print(f"FEVER JSONL not found: {fever_jsonl}")
            else:
                print('No FEVER gold path provided; skipping official FEVER eval.')
        except Exception as e:
            print('FEVER official eval failed:', e)

        # HotPotQA (distractor + fullwiki)
        try:
            for cfg, gold_path in [("distractor", args.hotpot_gold_distractor), ("fullwiki", args.hotpot_gold_fullwiki)]:
                if not gold_path:
                    print(f'No HotPotQA gold for {cfg}; skipping')
                    continue
                hp_ds_name = f"hotpotqa/hotpot_qa__{cfg}"
                hp_jsonl = out_dir / f"{safe_filename(hp_ds_name)}_outputs.jsonl"
                if hp_jsonl.exists():
                    cmd = [sys.executable, str(Path(__file__).parent / 'tools' / 'run_hotpot_eval.py'), '--jsonl', str(hp_jsonl), '--gold', gold_path, '--out_dir', str(official_out_dir / f'hotpot_{cfg}')]
                    print(f'Running HotPotQA official eval ({cfg})...')
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    open(official_out_dir / f'hotpot_{cfg}_stdout.txt', 'w', encoding='utf-8').write(p.stdout + '\n' + p.stderr)
                    # try parse metrics.json written by the hotpot runner
                    metrics_path = official_out_dir / f'hotpot_{cfg}' / 'hotpot_metrics.json'
                    metrics = None
                    if metrics_path.exists():
                        try:
                            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                        except Exception:
                            metrics = None
                    entry = {'dataset': hp_ds_name, 'official_eval_stdout_file': str(official_out_dir / f'hotpot_{cfg}_stdout.txt')}
                    if metrics is not None:
                        entry['official_metrics'] = metrics
                    else:
                        entry['official_metrics_json'] = str(metrics_path)
                    summary_list.append(entry)
                else:
                    print(f"HotPotQA JSONL not found for {cfg}: {hp_jsonl}")
        except Exception as e:
            print('HotPotQA official eval failed:', e)

        # Natural Questions
        try:
            if args.nq_gold:
                nq_preds = out_dir / 'nq_official_predictions.json'
                if nq_preds.exists():
                    cmd = [sys.executable, str(Path(__file__).parent / 'evals' / 'nq_eval.py'), '--gold_path', args.nq_gold, '--predictions_path', str(nq_preds)]
                    print('Running Natural Questions official eval...')
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    open(official_out_dir / 'nq_stdout.txt', 'w', encoding='utf-8').write(p.stdout + '\n' + p.stderr)
                    # try parse stdout for a JSON summary or leave raw
                    nq_metrics = None
                    try:
                        import re
                        jmatch = re.search(r"\{.*\}", p.stdout, flags=re.S)
                        if jmatch:
                            nq_metrics = json.loads(jmatch.group(0))
                    except Exception:
                        nq_metrics = None
                    entry = {'dataset': 'google-research-datasets/natural_questions', 'official_eval_stdout_file': str(official_out_dir / 'nq_stdout.txt')}
                    if nq_metrics is not None:
                        entry['official_metrics'] = nq_metrics
                    summary_list.append(entry)
                else:
                    print(f'No NQ official predictions file found at {nq_preds}; skipping NQ official eval.')
            else:
                print('No NQ gold path provided; skipping official NQ eval.')
        except Exception as e:
            print('NQ official eval failed:', e)

        # RAGTruth (optional docker runner)
        try:
            if args.ragtruth_model:
                cmd = [sys.executable, str(Path(__file__).parent / 'tools' / 'run_ragtruth_eval.py'), '--model_path', args.ragtruth_model, '--tokenizer', args.ragtruth_tokenizer]
                print('Running RAGTruth docker+predict_and_evaluate runner...')
                p = subprocess.run(cmd, capture_output=True, text=True)
                open(official_out_dir / 'ragtruth_stdout.txt', 'w', encoding='utf-8').write(p.stdout + '\n' + p.stderr)
                summary_list.append({
                    'dataset': 'wandb/RAGTruth-processed',
                    'official_eval_stdout_file': str(official_out_dir / 'ragtruth_stdout.txt')
                })
            else:
                print('No ragtruth_model provided; skipping RAGTruth docker-run.')
        except Exception as e:
            print('RAGTruth runner failed:', e)

        # Rewrite summary.json to include official eval pointers
        with open(summary_json, 'w', encoding='utf-8') as fh:
            json.dump(summary_list, fh, indent=2)
        print('Official evaluations finished; outputs saved to', str(official_out_dir))

if __name__ == "__main__":
    main()
