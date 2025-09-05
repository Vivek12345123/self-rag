"""
Small runner to convert existing TriviaQA JSONL outputs (produced by `run_eval.py`) into
the prediction JSON expected by the official evaluator and run it.

Usage:
  python tools/run_trivia_eval.py --jsonl path/to/mandarjoshi_trivia_qa__rc_outputs.jsonl \
      --dataset path/to/triviaqa-dev.json --out_dir ./trivia_eval_out

If `--dataset` is not provided, the script will attempt to download/locate the HF dataset
or require the user to provide the official TriviaQA JSON file.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def jsonl_to_pred_map(jsonl_path: Path) -> dict:
    preds = {}
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            # try to find a question id in the object; common keys: 'qid', 'id', 'index'
            qid = None
            for k in ('qid', 'id', 'example_id', 'index'):
                if k in obj:
                    qid = obj[k]
                    break
            if qid is None:
                # fallback: use the line index in file
                qid = str(len(preds))
            preds[str(qid)] = obj.get('prediction') or obj.get('prediction_text') or obj.get('prediction_raw') or obj.get('prediction') or obj.get('pred') or obj.get('prediction') or obj.get('prediction') or obj.get('predicted_answer') or obj.get('prediction') or obj.get('prediction') or obj.get('prediction')
            # As additional fallback, try 'prediction' or 'prediction_text' or 'prediction_raw'
            if not preds[str(qid)]:
                for alt in ('prediction', 'prediction_text', 'prediction_raw', 'pred', 'predicted_answer', 'prediction'):
                    if alt in obj and obj[alt]:
                        preds[str(qid)] = obj[alt]
                        break
            # final fallback: try 'instruction' (the question) as predicted answer (won't match golds)
            if not preds[str(qid)]:
                preds[str(qid)] = obj.get('instruction') or ''
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True, help='Path to JSONL outputs from run_eval.py for TriviaQA')
    parser.add_argument('--dataset', required=True, help='Path to official TriviaQA dataset JSON (downloaded)')
    parser.add_argument('--out_dir', default='./trivia_eval_out')
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_map = jsonl_to_pred_map(jsonl_path)
    pred_file = out_dir / 'predictions.json'
    with open(pred_file, 'w', encoding='utf-8') as fh:
        json.dump(pred_map, fh, indent=2, ensure_ascii=False)

    # Run the official evaluator we added
    eval_script = Path(__file__).parent.parent / 'evals' / 'triviaqa_eval.py'
    cmd = [sys.executable, str(eval_script), '--dataset_file', str(dataset_path), '--prediction_file', str(pred_file)]
    print('Running evaluator:', ' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    # write stdout/stderr for inspection
    with open(out_dir / 'trivia_stdout.txt', 'w', encoding='utf-8') as fh:
        fh.write(p.stdout + '\n' + p.stderr)
    # attempt to create a small metrics.json if the evaluator printed JSON-like summary
    metrics = {}
    # naive parse: look for lines like '"metric": value' in stdout
    try:
        # try to find a JSON object in stdout
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
    with open(out_dir / 'trivia_metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
