import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mdagents_sidechannel import call_log, upstream

# Upstream only recognises its own model names; the real backend is chosen by
# instrumentation.Config (MDA_MODEL / MDA_BASE_URL). See README.
UPSTREAM_MODEL = 'gpt-4o-mini'

# Fresh, dedicated CSV/text dir for this capture-correlation run, separate
# from smoke_test_logging.py's artifacts.
call_log.SMOKETEST_LOG_PATH = 'runs/logs/smoketest_log_pcaptest.csv'
call_log.TEXT_LOG_DIR = 'runs/logs/text_pcaptest'

utils = upstream.load()

if os.path.isfile(call_log.SMOKETEST_LOG_PATH):
    os.remove(call_log.SMOKETEST_LOG_PATH)

call_log.set_log_context(question_id='pcaptest', tier='pcaptest')

prompts = [
    "Reply with exactly one word: OK",
    "List 5 common fruits, comma separated, nothing else.",
    "In exactly 3 sentences, explain why the sky appears blue during the day.",
    "Name the capital of France in one word.",
    "In exactly 8 sentences, describe how a bicycle works, covering gears, brakes, and balance.",
]

print("Starting 5 calls -- make sure the packet capture is already running.")
for i, p in enumerate(prompts):
    agent = utils.Agent(instruction='You are a helpful assistant. Follow the length instruction exactly.',
                         role=f'pcaptest-{i}', model_info=UPSTREAM_MODEL)
    agent.chat(p)

with open(call_log.SMOKETEST_LOG_PATH, newline='') as f:
    rows = list(csv.DictReader(f))
print(f"\nDone. {len(rows)} calls logged to {call_log.SMOKETEST_LOG_PATH}.")
print(f"call_start_ts range: {rows[0]['call_start_ts']} .. {rows[-1]['call_end_ts']}")
print("Stop the capture (Ctrl-C in the tcpdump terminal) now, then hand the pcap path back.")
