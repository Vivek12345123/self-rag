#!/usr/bin/env python3
"""
Runner to compute official FEVER scores using fever.scorer.fever_score.

Inputs:
  --jsonl : path to our generated FEVER outputs JSONL (e.g. ./fever_outputs.jsonl)
  --gold  : path to FEVER gold file (dev.jsonl) in FEVER format
  --out_dir : directory to write stdout and JSON metrics

Outputs:
  - Writes a text file with printed metrics to <out_dir>/fever_stdout.txt
  - Writes a JSON file with numeric metrics to <out_dir>/fever_metrics.json

This script expects the `fever` package (fever.scorer) to be importable. If it's not
available, it will exit with a helpful message.
"""
import argparse
import json
import sys
from pathlib import Path


def load_gold(gold_path):
    actual = []
    with open(gold_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            # FEVER gold normally contains 'label' and 'evidence'
            if 'evidence' in j and j['evidence'] is not None:
                actual.append({'label': j.get('label'), 'evidence': j.get('evidence')})
            elif 'all_evidence' in j and j['all_evidence'] is not None:
                # convert each evidence tuple-like into [doc, sent]
                # user-provided dumps sometimes store entries like [ann_id, something, doc, sent, ...]
                ev = []
                for e in j.get('all_evidence'):
                    try:
                        # prefer indices 2 and 3 as in many FEVER preprocessed dumps
                        doc = e[2]
                        sent = e[3]
                        if doc is not None:
                            ev.append([doc, sent])
                    except Exception:
                        # fallback: if item is a list of two
                        if isinstance(e, (list, tuple)) and len(e) >= 2:
                            ev.append([e[0], e[1]])
                actual.append({'label': j.get('label'), 'evidence': ev})
            else:
                # Best-effort fallback: include label and empty evidence
                actual.append({'label': j.get('label'), 'evidence': j.get('evidence', [])})
    return actual


def load_predictions(pred_jsonl):
    preds = []
    with open(pred_jsonl, 'r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            j = json.loads(line)
            # Our evaluate_fever writes fields 'pred_label' and 'pred_evidence'
            label = j.get('pred_label') or j.get('predicted_label') or j.get('predicted') or None
            ev = j.get('pred_evidence') or j.get('predicted_evidence') or []
            # normalize evidence into list-of-lists [doc, sent]
            norm_ev = []
            for e in ev:
                if e is None:
                    continue
                if isinstance(e, dict):
                    # dict with keys like 'doc' and 'sent' or 'page' and 'sentence'
                    doc = e.get('doc') or e.get('page') or e.get('title') or e.get('wiki')
                    sent = e.get('sent') or e.get('sentence') or e.get('sent_idx') or e.get('s')
                    norm_ev.append([doc, sent])
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    norm_ev.append([e[0], e[1]])
                else:
                    # attempt to parse string patterns like Title#3
                    s = str(e)
                    if '#' in s:
                        parts = s.split('#')
                        try:
                            norm_ev.append([parts[0], int(parts[1])])
                        except Exception:
                            norm_ev.append([parts[0], parts[1] if len(parts) > 1 else None])
                    else:
                        norm_ev.append([s, None])
            preds.append({'predicted_label': label, 'predicted_evidence': norm_ev})
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True, help='Path to FEVER outputs JSONL produced by run_eval.py')
    parser.add_argument('--gold', required=True, help='Path to FEVER gold dev.jsonl')
    parser.add_argument('--out_dir', default='.', help='Directory to write outputs')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from fever.scorer import fever_score
    except Exception as e:
        print('Could not import fever.scorer. Please install the FEVER scorer package.', file=sys.stderr)
        print('Import error:', e, file=sys.stderr)
        sys.exit(2)

    golds = load_gold(args.gold)
    preds = load_predictions(args.jsonl)

    try:
        score, acc, pr, rec, f1 = fever_score(preds, golds, max_evidence=None)
    except Exception as e:
        print('fever_score failed:', e, file=sys.stderr)
        sys.exit(3)

    out_text = f"FEVER official results:\nscore={score}\nacc={acc}\nprecision={pr}\nrecall={rec}\nf1={f1}\n"
    # write stdout-equivalent and JSON
    with open(out_dir / 'fever_stdout.txt', 'w', encoding='utf-8') as fh:
        fh.write(out_text)

    with open(out_dir / 'fever_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump({'score': score, 'accuracy': acc, 'precision': pr, 'recall': rec, 'f1': f1}, fh, indent=2)

    print(out_text)


if __name__ == '__main__':
    main()
