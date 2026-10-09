"""Build a class-balanced DDXPlus sample for the diagnosis-leakage experiment.

The earlier ddxplus run (data/ddxplus/test.jsonl, 300 rows -- itself a
small `--max-test` slice prepare_ddxplus.py took off the front of the raw
release) produced 45 distinct diagnoses over 220 completed cases: ~5
cases/pathology on average, far too few to train or evaluate a 45-way
classifier (see runs/results/paper/ -- macro-F1 there was
indistinguishable from chance). This script samples fresh, directly from
the full raw release (--raw-dir, ~134k rows, 49 pathologies), keeps only
the --n-classes best-represented pathologies, and draws exactly
--per-class cases from each -- so the resulting classification target has
a real, known, BALANCED base rate instead of DDXPlus's natural long tail.

Question/option construction (build_record/describe_evidences below) is
copied from prepare_ddxplus.py rather than imported from it -- that
script builds its argparse `args` at module import time from its own
flags, so importing it here would crash on this script's different CLI
arguments the moment argparse hit an unrecognized flag.

Usage (from the repository root):
    python scripts/sample_ddxplus_balanced.py --raw-dir ../DDXPlus_data \
        --n-classes 12 --per-class 50 --seed 42

Writes:
    data/ddxplus_balanced/test.jsonl   -- 600 records in main.py's expected
                                           schema (question/options/answer/
                                           answer_idx). Run with:
                                               python main.py --dataset ddxplus_balanced
    data/ddxplus_balanced/train.jsonl  -- copied as-is from --train-source
                                           (few-shot exemplars; balance
                                           doesn't matter for these)
    runs/samples/ddxplus_balanced_<seed>.csv
                                        -- audit trail (balanced_question_id,
                                           pathology, source_row_index, age,
                                           sex) so the exact sample is
                                           reproducible and traceable back
                                           to the raw release.
"""
import argparse
import ast
import csv
import json
import os
import random
import shutil
import string

parser = argparse.ArgumentParser()
parser.add_argument('--raw-dir', default='../DDXPlus_data')
parser.add_argument('--n-classes', type=int, default=12)
parser.add_argument('--per-class', type=int, default=50)
parser.add_argument('--max-options', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out-dataset-dir', default='data/ddxplus_balanced')
parser.add_argument('--sample-out-dir', default='runs/samples')
parser.add_argument('--train-source', default='data/ddxplus/train.jsonl',
                     help='few-shot exemplars to reuse as-is (balance not required for these)')
args = parser.parse_args()


# ---- copied from prepare_ddxplus.py (see module docstring for why) --------

def load_evidences(raw_dir):
    with open(os.path.join(raw_dir, 'release_evidences.json'), encoding='utf-8') as f:
        return json.load(f)


def describe_evidences(codes, evidences):
    grouped = {}
    order = []
    for code in codes:
        base, _, val = code.partition('_@_')
        if base not in grouped:
            grouped[base] = []
            order.append(base)
        if val:
            if val.startswith('V_'):
                meaning = evidences.get(base, {}).get('value_meaning', {}).get(val)
                grouped[base].append(meaning['en'] if meaning else val)
            else:
                grouped[base].append(val)
        else:
            grouped[base].append('Yes')

    lines = []
    for base in order:
        info = evidences.get(base)
        if info is None:
            continue
        question = info['question_en'].rstrip('?:').strip()
        values = grouped[base]
        if values == ['Yes']:
            lines.append(f"{question}: Yes")
        else:
            lines.append(f"{question}: {', '.join(values)}")
    return lines


def build_record(row, evidences, max_options, rng):
    age = row['AGE']
    sex = 'male' if row['SEX'] == 'M' else 'female'
    evidence_codes = ast.literal_eval(row['EVIDENCES'])
    initial_code = row['INITIAL_EVIDENCE']
    pathology = row['PATHOLOGY']
    differential = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])

    initial_lines = describe_evidences([initial_code], evidences)
    chief_complaint = initial_lines[0] if initial_lines else initial_code

    other_codes = [c for c in evidence_codes if c.partition('_@_')[0] != initial_code.partition('_@_')[0]]
    finding_lines = describe_evidences(other_codes, evidences)

    question = (
        f"A {age}-year-old {sex} patient presents to the clinic.\n"
        f"Chief complaint: {chief_complaint}\n\n"
        "During evaluation, the following was noted:\n"
        + "\n".join(f"- {line}" for line in finding_lines)
    )

    names = [name for name, _ in differential]
    if pathology not in names:
        names.append(pathology)
    kept = names[:max_options]
    if pathology not in kept:
        kept[-1] = pathology
    rng.shuffle(kept)

    letters = list(string.ascii_uppercase[:len(kept)])
    options = dict(zip(letters, kept))
    answer_idx = letters[kept.index(pathology)]

    return {
        'question': question,
        'options': options,
        'answer': pathology,
        'answer_idx': answer_idx,
    }


# ---- sampling ---------------------------------------------------------

def main():
    test_path = os.path.join(args.raw_dir, 'release_test_patients')
    print(f"Reading {test_path} ...")
    by_pathology = {}
    with open(test_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row['_source_row_index'] = idx
            by_pathology.setdefault(row['PATHOLOGY'], []).append(row)
    total_rows = sum(len(v) for v in by_pathology.values())
    print(f"{total_rows} rows, {len(by_pathology)} distinct pathologies in the full source pool")
    print()

    counts = {name: len(rows) for name, rows in by_pathology.items()}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:args.n_classes]

    print(f"Top {args.n_classes} pathologies by availability (full pool):")
    for name, cnt in top:
        print(f"  {cnt:6d}  {name}")
    print()

    too_small = [(name, cnt) for name, cnt in top if cnt < args.per_class]
    if too_small:
        raise SystemExit(f"Not enough source cases for: {too_small} (need >= {args.per_class} each)")

    rng = random.Random(args.seed)
    sampled_rows = []
    for name, _ in top:  # fixed rank order -> reproducible given the same seed
        chosen = rng.sample(by_pathology[name], args.per_class)
        sampled_rows.extend(chosen)
    rng.shuffle(sampled_rows)  # interleave classes so dataset position doesn't itself leak the label

    class_counts = {}
    for r in sampled_rows:
        class_counts[r['PATHOLOGY']] = class_counts.get(r['PATHOLOGY'], 0) + 1
    print(f"Sampled {len(sampled_rows)} cases across {len(top)} classes, seed={args.seed}")
    print("Resulting class distribution (post-sample):")
    for name, cnt in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cnt:3d}  {name}")
    chance = 1.0 / len(top)
    print(f"chance floor (1/{len(top)}): {chance:.4f}")
    print()

    evidences = load_evidences(args.raw_dir)
    os.makedirs(args.out_dataset_dir, exist_ok=True)
    os.makedirs(args.sample_out_dir, exist_ok=True)

    test_out = os.path.join(args.out_dataset_dir, 'test.jsonl')
    audit_out = os.path.join(args.sample_out_dir, f'ddxplus_balanced_{args.seed}.csv')

    with open(test_out, 'w', encoding='utf-8') as jf, open(audit_out, 'w', newline='') as af:
        writer = csv.DictWriter(af, fieldnames=['balanced_question_id', 'pathology', 'source_row_index', 'age', 'sex'])
        writer.writeheader()
        for qid, row in enumerate(sampled_rows):
            record = build_record(row, evidences, args.max_options, rng)
            jf.write(json.dumps(record) + '\n')
            writer.writerow({
                'balanced_question_id': qid,
                'pathology': row['PATHOLOGY'],
                'source_row_index': row['_source_row_index'],
                'age': row['AGE'],
                'sex': row['SEX'],
            })

    train_out = os.path.join(args.out_dataset_dir, 'train.jsonl')
    shutil.copyfile(args.train_source, train_out)

    print(f"Wrote {len(sampled_rows)} records to {test_out}")
    print(f"Wrote few-shot exemplars to {train_out} (copied from {args.train_source})")
    print(f"Wrote audit trail to {audit_out}")


if __name__ == '__main__':
    main()
