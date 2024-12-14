import os
import json
import argparse
import pandas as pd


def read_results(
    file_path: str,
    task_name: str,
    metric: str,
):
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = data['results'][task_name][metric]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--metric', type=str, default='exact_match,strict-match')
    args = parser.parse_args()

    acc_dirs = [d for d in os.listdir(args.file_path) if d.endswith('accuracy')]
    acc_dirs = [os.path.join(args.file_path, d) for d in acc_dirs]
    acc_dirs = [os.path.join(d, os.listdir(d)[0]) for d in acc_dirs]
    acc_files = [os.path.join(d, f) for d in acc_dirs for f in os.listdir(d) if f.endswith('.json')]
    results = [read_results(f, args.task_name, args.metric) for f in acc_files]
    results = pd.DataFrame(results)
    print(results)
