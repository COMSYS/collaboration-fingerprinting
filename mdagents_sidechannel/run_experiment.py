"""Run MDAgents over a dataset while logging every API call.

Replaces upstream's `main.py`, which we neither modify nor ship. The loop here
calls upstream's public functions in the only order they can be called; the
additions are ours: a resumable start index, per-question log context so each
call can be attributed to a case and a tier, and output paths under runs/.

Upstream branches on the literal dataset name 'medqa' in three places, and
the branch is NOT cosmetic -- it changes both the prompts and the number of
API calls:

  * create_question appends " Options: (A) ... (B) ..." only for 'medqa'.
  * process_basic_query generates five few-shot examplers only for 'medqa',
    each one an extra API call (basic cases: 9 calls with it, 4 without).
  * process_intermediate_query builds few-shot text only for 'medqa'.

The published runs therefore did NOT all take the same path, and reproducing
them means reproducing that difference exactly:

  medqa            -> presented to upstream as 'medqa'     (examplers, options)
  ddxplus          -> presented as 'medqa'                 (matches the legacy run)
  ddxplus_balanced -> presented as itself, i.e. NOT 'medqa' (no examplers, no
                      options -- verified against the collected prompt text)

Use --upstream-dataset to override. Upstream's own load_data() is not used --
it resolves '../data/...' relative to the process working directory, which we
do not want to depend on.
"""
import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from . import upstream
from .call_log import set_log_context

# How each of our dataset names was presented to upstream in the published
# runs. Anything not listed passes through unchanged, which means upstream
# takes its non-'medqa' path: no options appended, no few-shot examplers.
UPSTREAM_ALIAS = {'medqa': 'medqa', 'ddxplus': 'medqa'}


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='medqa',
                   help="dataset directory under --data-dir (real name, e.g. ddxplus_balanced)")
    p.add_argument('--model', default='gpt-4o-mini',
                   help="model name handed to upstream; the actual backend is set by "
                        "MDA_MODEL/MDA_BASE_URL (see instrumentation.Config)")
    p.add_argument('--difficulty', default='adaptive')
    p.add_argument('--num_samples', type=int, default=1400)
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='runs/output')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--upstream-dataset', default=None,
                   help="name handed to upstream's dataset branches; defaults to "
                        "the mapping that reproduces the published runs")
    args = p.parse_args()

    random.seed(args.seed)

    utils = upstream.load()

    upstream_dataset = args.upstream_dataset or UPSTREAM_ALIAS.get(args.dataset, args.dataset)
    takes_medqa_path = upstream_dataset == 'medqa'
    print(f"[run] upstream sees dataset={upstream_dataset!r}: "
          f"{'options appended, few-shot examplers generated' if takes_medqa_path else 'no options, no few-shot examplers'}")

    data_dir = Path(args.data_dir) / args.dataset
    test_qa = load_jsonl(data_dir / 'test.jsonl')
    examplers = load_jsonl(data_dir / 'train.jsonl')
    test_qa = test_qa[args.start_index:args.start_index + args.num_samples]
    print(f"[run] {args.dataset}: {len(test_qa)} questions "
          f"(from index {args.start_index}), {len(examplers)} examplers")

    # Upstream reads only `.dataset` off this object.
    upstream_args = SimpleNamespace(dataset=upstream_dataset, model=args.model,
                                    difficulty=args.difficulty)

    results = []
    for no, sample in enumerate(tqdm(test_qa), start=args.start_index):
        print(f"\n[INFO] no: {no}")
        # Tier is unknown until determine_difficulty returns, so the
        # difficulty-determination call itself is logged with tier=None.
        set_log_context(question_id=no, tier=None)

        question, img_path = utils.create_question(sample, upstream_dataset)
        difficulty = utils.determine_difficulty(question, args.difficulty)
        set_log_context(tier=difficulty)
        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision = utils.process_basic_query(
                question, examplers, args.model, upstream_args)
        elif difficulty == 'intermediate':
            final_decision = utils.process_intermediate_query(
                question, examplers, args.model, upstream_args)
        elif difficulty == 'advanced':
            final_decision = utils.process_advanced_query(
                question, args.model, upstream_args)
        else:
            print(f"[WARN] unrecognised difficulty {difficulty!r}; skipping")
            continue

        results.append({
            'question': question,
            'label': sample.get('answer_idx'),
            'answer': sample.get('answer'),
            'options': sample.get('options'),
            'response': final_decision,
            'difficulty': difficulty,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = os.environ.get('MDA_MODEL', 'gpt-oss-120b')
    out_path = out_dir / f"{backend}_{args.dataset}_{args.difficulty}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n[run] wrote {len(results)} results to {out_path}")


if __name__ == '__main__':
    main()
