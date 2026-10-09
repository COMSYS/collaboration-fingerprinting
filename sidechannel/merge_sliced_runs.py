"""Merge multiple slice_pcap.py runs (e.g. a dataset run split into halves,
or resumed after a dropped capture) into one sliced CSV with latency baked
in -- safe to concatenate, unlike joining on call_index across runs.

Why this needs to exist: utils.py's _call_index counter resets to 0 in
every fresh process (see utils.py:24), so two separate main.py/
run_ddxplus_balanced.py invocations each produce a log CSV and a
sliced.csv with their own call_index sequences starting at 1 --
call_index=1 exists once per run, not once globally. Naively
concatenating two sliced.csv files and then joining the result against a
concatenation of their log CSVs on call_index (e.g. to pull in
latency_seconds, as diagnosis_classifier.py/tier_classifier.py's
--log-csv does) silently produces a many-to-many join and corrupts the
data. The fix: join each run's sliced.csv against its OWN log CSV (where
call_index is genuinely unique) first, THEN concatenate -- which is
exactly what this script does. question_id is unaffected by any of this
(main.py assigns it via --start_index, so it's globally unique across
runs by construction, unlike call_index).

Usage (from the repository root):
    python sidechannel/merge_sliced_runs.py \
        --run runs/captures/ddxplus_balanced_full_1_sliced.csv:runs/logs/ddxplus_balanced_full_1/smoketest_log.csv \
        --run runs/captures/ddxplus_balanced_full_2_sliced.csv:runs/logs/ddxplus_balanced_full_2/smoketest_log.csv \
        --out runs/traces/ddxplus_balanced/wire-sliced.csv
"""
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='append', required=True, dest='runs',
                     metavar='SLICED_CSV:LOG_CSV',
                     help='one per run; repeatable. Each pair is joined on call_index '
                          '(unique within that run) before any concatenation happens.')
parser.add_argument('--out', required=True)
args = parser.parse_args()


def main():
    all_rows = []
    fieldnames = None
    all_qid_sets = []
    for i, pair in enumerate(args.runs):
        sliced_path, log_path = pair.split(':', 1)

        with open(log_path, newline='') as f:
            latency_by_call_index = {row['call_index']: row['latency_seconds'] for row in csv.DictReader(f)}

        with open(sliced_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if fieldnames is None:
                fieldnames = reader.fieldnames + ['latency_seconds', 'source_run']

        missing = 0
        for row in rows:
            latency = latency_by_call_index.get(row['call_index'])
            if latency is None:
                missing += 1
            row['latency_seconds'] = latency if latency is not None else ''
            row['source_run'] = sliced_path
        if missing:
            print(f"WARNING: {missing}/{len(rows)} rows in {sliced_path} had no matching "
                  f"call_index in {log_path}")

        run_qids = {int(r['question_id']) for r in rows}
        print(f"run {i} ({sliced_path}): {len(rows)} rows ({len(run_qids)} cases), "
              f"question_id range {min(run_qids)}-{max(run_qids)}")
        all_rows.extend(rows)
        all_qid_sets.append(run_qids)

    # A question_id naturally repeats many times *within* one run (once per
    # call belonging to that case) -- that's normal, not a collision. What
    # actually matters is whether the same question_id shows up in more
    # than one *run*, which would mean the runs weren't actually disjoint.
    overlap = set()
    for a in range(len(all_qid_sets)):
        for b in range(a + 1, len(all_qid_sets)):
            overlap |= all_qid_sets[a] & all_qid_sets[b]
    if overlap:
        print(f"WARNING: {len(overlap)} question_id(s) appear in more than one run -- "
              f"these runs are NOT disjoint: {sorted(overlap)[:10]}"
              f"{' ...' if len(overlap) > 10 else ''}")
    else:
        print(f"Confirmed: all {sum(len(s) for s in all_qid_sets)} cases across "
              f"{len(all_qid_sets)} runs are disjoint (no question_id overlap).")

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} merged rows ({len(args.runs)} runs) to {args.out}")


if __name__ == '__main__':
    main()
