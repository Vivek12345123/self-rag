"""
Convert JSONL outputs from `run_eval.py` for HotPotQA into the prediction format expected by `evals/hotpot_eval.py`
and run the evaluator.

Assumes the JSONL lines contain at least: 'index' (or 'id'/'qid'), 'prediction' (answer string), and optionally
'predicted_supporting_facts' or 'sp' as a list of [title, sent_idx] pairs. Falls back to parsing 'prediction' for answer.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def jsonl_to_hotpot_pred(jsonl_path: Path) -> dict:
    preds = {'answer': {}, 'sp': {}}
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = None
            for k in ('qid', 'id', 'index', 'example_id'):
                if k in obj:
                    qid = str(obj[k])
                    break
            if qid is None:
                qid = str(i)

            # Try to get the predicted answer text
            answer = obj.get('prediction') or obj.get('prediction_text') or obj.get('prediction_raw') or obj.get('pred') or obj.get('predicted_answer') or obj.get('prediction') or ''
            preds['answer'][qid] = answer

            # supporting facts
            sp = obj.get('predicted_supporting_facts') or obj.get('sp') or obj.get('pred_sp') or None
            if sp is None:
                # attempt to extract '(Title, idx)' patterns from raw text
                raw = obj.get('prediction_raw') or obj.get('model_raw') or obj.get('prediction') or ''
                # simple regex: look for Title#3 patterns
                import re
                found = re.findall(r"([A-Za-z0-9 _\-/]+?)#(\d+)", raw)
                if found:
                    sp_list = [[t.strip(), int(i)] for t, i in found]
                else:
                    sp_list = []
            else:
                sp_list = sp
            preds['sp'][qid] = sp_list

    return preds


def run_eval(jsonl_path, gold_file, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_map = jsonl_to_hotpot_pred(Path(jsonl_path))
    pred_file = out_dir / 'hotpot_predictions.json'
    with open(pred_file, 'w', encoding='utf-8') as fh:
        json.dump(pred_map, fh, indent=2, ensure_ascii=False)

    # Run the hotpot evaluator
    eval_script = Path(__file__).parent.parent / 'evals' / 'hotpot_eval.py'
    cmd = [sys.executable, str(eval_script), str(pred_file), str(gold_file)]
    print('Running hotpot evaluator:', ' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    # capture outputs
    (out_dir / 'hotpot_stdout.txt').write_text(p.stdout + '\n' + p.stderr, encoding='utf-8')
    # Try to parse any JSON summary from stdout, otherwise save raw
    metrics = {}
    try:
        import re
        jmatch = re.search(r"\{.*\}", p.stdout, flags=re.S)
        if jmatch:
            try:
                metrics = json.loads(jmatch.group(0))
            except Exception:
                metrics = {"raw_stdout": p.stdout}
        else:
            metrics = {"raw_stdout": p.stdout}
    except Exception:
        metrics = {"raw_stdout": p.stdout}
    with open(out_dir / 'hotpot_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--out_dir', default='./hotpot_eval_out')
    args = parser.parse_args()
    run_eval(args.jsonl, args.gold, args.out_dir)


if __name__ == '__main__':
    main()
