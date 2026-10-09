"""Join a sliced.csv (from slice_pcap.py) with the dataset's ground-truth
answers, keyed by question_id -- the 0-based line index into
data/<dataset>/test.jsonl that main.py assigns via set_log_context
(see main.py:34 and utils.py's load_data/create_question).

Usage (from the repository root):
    python sidechannel/enrich_sliced.py --sliced-csv runs/captures/sliced.csv \
        --dataset medqa --out runs/captures/sliced_enriched.csv

question_id values that aren't a valid dataset row index (e.g. "pcaptest"
from a smoke test run) are left blank rather than causing a crash.
"""
import argparse
import csv
import json

parser = argparse.ArgumentParser()
parser.add_argument('--sliced-csv', required=True)
parser.add_argument('--dataset', required=True, help='must have data/<dataset>/test.jsonl')
parser.add_argument('--out', default=None, help='defaults to <sliced-csv>_enriched.csv')
args = parser.parse_args()


def main():
    test_path = f'data/{args.dataset}/test.jsonl'
    with open(test_path) as f:
        test_qa = [json.loads(line) for line in f]

    with open(args.sliced_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    out_path = args.out or args.sliced_csv.rsplit('.csv', 1)[0] + '_enriched.csv'

    missing = 0
    for row in rows:
        qid = row['question_id']
        sample = test_qa[int(qid)] if qid.isdigit() and int(qid) < len(test_qa) else None
        if sample is None:
            missing += 1
            row['answer'] = ''
            row['answer_idx'] = ''
        else:
            row['answer'] = sample['answer']
            row['answer_idx'] = sample['answer_idx']

    if missing:
        print(f"WARNING: {missing}/{len(rows)} rows had a question_id that isn't a valid "
              f"index into {test_path} (e.g. a non-numeric ID from a smoke test run) -- "
              f"left answer/answer_idx blank for those.")

    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
