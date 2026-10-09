"""Convert the raw DDXPlus release CSVs into the medqa-style jsonl schema
that main.py / utils.py (load_data, create_question) expect at
data/<dataset>/{train,test}.jsonl.

Each output record looks like:
{
    "question": "<clinical vignette built from age/sex/evidences>",
    "options": {"A": "Bronchitis", "B": "GERD", ...},
    "answer": "GERD",
    "answer_idx": "B"
}

Usage (from the MDAgents/ directory):
    python prepare_ddxplus.py --raw-dir ../DDXPlus_data
"""
import argparse
import ast
import csv
import io
import json
import os
import random
import string
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--raw-dir', type=str, default='../DDXPlus_data',
                     help='Directory containing the release_* DDXPlus files')
parser.add_argument('--out-dir', type=str, default='data/ddxplus')
parser.add_argument('--max-train', type=int, default=500,
                     help='Number of train rows to convert (used only for few-shot exemplars)')
parser.add_argument('--max-test', type=int, default=1000,
                     help='Number of test rows to convert')
parser.add_argument('--max-options', type=int, default=10,
                     help='Cap on multiple-choice options per question (ground truth is always kept)')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)


def load_evidences(raw_dir):
    with open(os.path.join(raw_dir, 'release_evidences.json'), encoding='utf-8') as f:
        return json.load(f)


def describe_evidences(codes, evidences):
    """Group evidence codes by base code and render each as a short clinical fact."""
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


def build_record(row, evidences, max_options):
    age = row['AGE']
    sex = 'male' if row['SEX'] == 'M' else 'female'
    evidence_codes = ast.literal_eval(row['EVIDENCES'])
    initial_code = row['INITIAL_EVIDENCE']
    pathology = row['PATHOLOGY']
    differential = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])  # [[name, prob], ...] sorted desc

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
    random.shuffle(kept)

    letters = list(string.ascii_uppercase[:len(kept)])
    options = dict(zip(letters, kept))
    answer_idx = letters[kept.index(pathology)]

    return {
        'question': question,
        'options': options,
        'answer': pathology,
        'answer_idx': answer_idx,
    }


def convert(rows_iter, evidences, limit, max_options, out_path):
    count = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for row in rows_iter:
            if count >= limit:
                break
            record = build_record(row, evidences, max_options)
            out.write(json.dumps(record) + '\n')
            count += 1
    return count


def main():
    evidences = load_evidences(args.raw_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    test_path = os.path.join(args.raw_dir, 'release_test_patients')
    with open(test_path, encoding='utf-8') as f:
        n_test = convert(csv.DictReader(f), evidences, args.max_test, args.max_options,
                          os.path.join(args.out_dir, 'test.jsonl'))
    print(f"Wrote {n_test} test records to {args.out_dir}/test.jsonl")

    train_zip = os.path.join(args.raw_dir, 'release_train_patients.zip')
    with zipfile.ZipFile(train_zip) as z:
        name = z.namelist()[0]
        with z.open(name) as raw:
            wrapped = io.TextIOWrapper(raw, encoding='utf-8')
            n_train = convert(csv.DictReader(wrapped), evidences, args.max_train, args.max_options,
                               os.path.join(args.out_dir, 'train.jsonl'))
    print(f"Wrote {n_train} train records to {args.out_dir}/train.jsonl")


if __name__ == '__main__':
    main()
