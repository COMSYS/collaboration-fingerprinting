import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mdagents_sidechannel import call_log, upstream

# Upstream only recognises its own model names; the real backend is chosen by
# instrumentation.Config (MDA_MODEL / MDA_BASE_URL). See README.
UPSTREAM_MODEL = 'gpt-4o-mini'

# Keep this smoke test's output separate from the real research log.
call_log.SMOKETEST_LOG_PATH = 'runs/logs/smoketest_log_test.csv'
call_log.TEXT_LOG_DIR = 'runs/logs/text_test'

utils = upstream.load()

if os.path.isfile(call_log.SMOKETEST_LOG_PATH):
    os.remove(call_log.SMOKETEST_LOG_PATH)

call_log.set_log_context(question_id='smoke', tier='smoke')

# Prompts sized to elicit *visible* completions of increasing length. A
# fixed "reply with one word" prompt can't demonstrate response_bytes
# scaling -- the answer never changes -- so length has to vary in what the
# model is asked to produce, not just in the request.
prompts = [
    "Reply with exactly one word: OK",
    "List 5 common fruits, comma separated, nothing else.",
    "In exactly 6 sentences, explain why the sky appears blue during the day.",
]

for i, p in enumerate(prompts):
    agent = utils.Agent(instruction='You are a helpful assistant. Follow the length instruction exactly.',
                         role=f'smoke-{i}', model_info=UPSTREAM_MODEL)
    agent.chat(p)

with open(call_log.SMOKETEST_LOG_PATH, newline='') as f:
    rows = list(csv.DictReader(f))

print(f"Logged {len(rows)} rows (expected 3).")
req_sizes = [int(r['request_bytes']) for r in rows]
print("request_bytes per call:", req_sizes)

assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
assert all(s > 0 for s in req_sizes), "request_bytes must be nonzero for every call"

for r in rows:
    assert r['prompt_text'] and os.path.isfile(r['prompt_text']), f"missing prompt_text file for row {r['call_index']}"
    assert r['completion_text'] and os.path.isfile(r['completion_text']), f"missing completion_text file for row {r['call_index']}"
    with open(r['prompt_text'], encoding='utf-8') as pf:
        recomputed = len(pf.read().encode('utf-8'))
    assert recomputed == int(r['request_bytes']), (
        f"row {r['call_index']}: stored request_bytes={r['request_bytes']} != "
        f"recomputed {recomputed} from saved prompt_text file"
    )
    assert r['call_start_ts'] == r['timestamp'], f"row {r['call_index']}: timestamp is not aliased to call_start_ts"
    assert int(r['call_end_ts']) > int(r['call_start_ts']), f"row {r['call_index']}: call_end_ts must be after call_start_ts"

print("\ncompletion_tokens vs response_bytes vs bytes/token:")
print(f"{'#':<3}{'completion_tokens':<20}{'response_bytes':<18}{'bytes/token':<12}{'completion_text'}")
for r in rows:
    ctok = int(r['completion_tokens'])
    rbytes = int(r['response_bytes'])
    with open(r['completion_text'], encoding='utf-8') as cf:
        ctext = cf.read()
    bpt = round(rbytes / ctok, 2) if ctok else float('nan')
    print(f"{r['call_index']:<3}{ctok:<20}{rbytes:<18}{bpt:<12}{ctext[:60]!r}")

print("\nOK: request_bytes nonzero; byte counts reproducible from saved prompt_text files; "
      "timestamp/call_start_ts aliased correctly. See table above for the "
      "completion_tokens vs response_bytes relationship -- inspect before trusting it.")
