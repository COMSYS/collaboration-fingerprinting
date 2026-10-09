"""Launch main.py against data/ddxplus_balanced with dedicated, non-colliding
log/text/output paths -- for the balanced-DDXPlus diagnosis-leakage run
(see scripts/sample_ddxplus_balanced.py for how that dataset was built).

main.py has no flag for the log path (it's hardcoded in utils.py via
SMOKETEST_LOG_PATH/TEXT_LOG_DIR), and it's a flat script with no
`if __name__ == '__main__':` guard, so it can't take a log-path argument
or be safely called as a function. Instead this monkey-patches
utils.SMOKETEST_LOG_PATH/TEXT_LOG_DIR *before* running main.py's body via
runpy -- the exact same pattern tests/smoke_test_logging.py and
tests/smoke_test_pcap.py already use, just applied to a full main.py run
instead of a handful of canned prompts. Nothing in utils.py's logging
mechanism itself (measured wire bytes, call_start_ts/call_end_ts
bracketing, flush-safe writes) is touched -- only *where* it writes.

Usage (from the repository root):
    python scripts/run_ddxplus_balanced.py --num_samples 3 --tag pilot
    python scripts/run_ddxplus_balanced.py --num_samples 600 --tag full

Writes:
    runs/logs/ddxplus_balanced_<tag>/smoketest_log.csv
    runs/logs/ddxplus_balanced_<tag>/text/
    runs/output/gpt-oss-120b_ddxplus_balanced_adaptive_<tag>.json  (renamed
        post-run from main.py's own hardcoded runs/output/{model}_{dataset}_
        {difficulty}.json -- that path has no --tag/--start_index in it, so
        two --tag runs against the same --dataset would otherwise silently
        overwrite each other's results JSON. Doesn't affect the wire-level
        side-channel analysis itself, which never reads this file -- only
        main.py's own accuracy bookkeeping.)
"""
import argparse
import os
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, required=True)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--model', default='gpt-oss-120b')
parser.add_argument('--difficulty', default='adaptive')
parser.add_argument('--tag', required=True, help='run label, e.g. "pilot" or "full" -- picks the log subdir')
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils  # noqa: E402  (import after argparse so --help doesn't pay utils' import cost)

log_dir = f'runs/logs/ddxplus_balanced_{args.tag}'
utils.SMOKETEST_LOG_PATH = os.path.join(log_dir, 'smoketest_log.csv')
utils.TEXT_LOG_DIR = os.path.join(log_dir, 'text')
print(f"[run_ddxplus_balanced] logging to {utils.SMOKETEST_LOG_PATH} / {utils.TEXT_LOG_DIR}/")

sys.argv = [
    'main.py',
    '--dataset', 'ddxplus_balanced',
    '--model', args.model,
    '--difficulty', args.difficulty,
    '--num_samples', str(args.num_samples),
    '--start_index', str(args.start_index),
]
print(f"[run_ddxplus_balanced] running: {' '.join(sys.argv)}")
runpy.run_path('main.py', run_name='__main__')

# main.py's own output path has no --tag/--start_index in it -- rename it
# immediately so a second --tag run against the same --dataset can't
# silently overwrite this run's results.
default_output = f'runs/output/{args.model}_ddxplus_balanced_{args.difficulty}.json'
tagged_output = f'runs/output/{args.model}_ddxplus_balanced_{args.difficulty}_{args.tag}.json'
if os.path.exists(default_output):
    os.rename(default_output, tagged_output)
    print(f"[run_ddxplus_balanced] renamed {default_output} -> {tagged_output}")
