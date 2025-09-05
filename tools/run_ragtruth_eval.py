"""
Runner to launch a TGI docker container hosting a local model and run predict_and_evaluate.py for RAGTruth.

Usage example:
  python tools/run_ragtruth_eval.py --model_path baseline --tokenizer meta-llama/Llama-2-13b-hf

This script only constructs and invokes the docker and python commands; it requires docker installed and
the `predict_and_evaluate.py` script available in the repo.
"""
import argparse
import subprocess
import shlex
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Local model directory under /data/exp/<model_path> to mount')
    parser.add_argument('--tokenizer', default='meta-llama/Llama-2-13b-hf')
    parser.add_argument('--docker_name', default='baseline')
    parser.add_argument('--gpu_device', default='7')
    parser.add_argument('--host_port', type=int, default=8300)
    parser.add_argument('--out_dir', default='.', help='Directory to write stdout/stderr outputs (optional)')
    args = parser.parse_args()

    model_path = args.model_path
    docker_name = args.docker_name
    gpu = args.gpu_device
    port = args.host_port
    tokenizer = args.tokenizer

    docker_cmd = f"docker run -d --name {docker_name} --gpus '\"device={gpu}\"' -v $PWD:/data --shm-size 1g -p {port}:80 ghcr.io/huggingface/text-generation-inference:2.0.1 --model-id /data/exp/{model_path} --dtype bfloat16 --max-total-tokens 8000 --sharded false --max-input-length 4095"
    print('Docker command to run:')
    print(docker_cmd)
    # run docker
    try:
        subprocess.run(docker_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print('Docker run failed:', e, file=sys.stderr)
        sys.exit(1)

    # now run predict_and_evaluate.py
    out_dir = args.out_dir
    py_cmd = f"python predict_and_evaluate.py --model_name {model_path} --tokenizer {tokenizer}"
    print('Running prediction and evaluation:')
    print(py_cmd)
    try:
        # capture stdout/stderr and write to file in out_dir
        p = subprocess.run(py_cmd, shell=True, capture_output=True, text=True)
        logfile = Path(out_dir) / 'ragtruth_stdout.txt'
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, 'w', encoding='utf-8') as fh:
            fh.write(p.stdout or '')
            fh.write('\n--- STDERR ---\n')
            fh.write(p.stderr or '')
        if p.returncode != 0:
            print('predict_and_evaluate returned non-zero exit code; see', str(logfile), file=sys.stderr)
            sys.exit(p.returncode)
    except subprocess.CalledProcessError as e:
        print('predict_and_evaluate failed:', e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
