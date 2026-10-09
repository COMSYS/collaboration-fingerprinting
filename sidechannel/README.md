# Side-channel tooling

Measures whether MDAgents' encrypted API traffic leaks information — which
difficulty tier handled a case, how long the answer was — from wire-level
metadata alone. No TLS is decrypted and no key material is touched: TLS record
headers (content type, version, length) are plaintext by design, so a capture
gives per-segment direction, size, and handshake-vs-application-data without
the session key.

`mdagents_sidechannel/call_log.py` records a nanosecond start/end window per API
call. `slice_pcap.py` joins that to the capture, summarising the packets inside
each window. The classifiers then ask whether the tier or the diagnosis is
recoverable from those summaries.

## Prerequisites

- `tshark` on `PATH` (reading a `.pcap` needs no root; live capture does)
- `sudo`, for `capture.sh`
- the collection setup from the top-level README: `requirements-collect.txt`,
  `./setup_mdagents.sh`, and `MDA_BASE_URL` plus the API key

## Pipeline

Run from the repository root.

**1. Capture**, in its own terminal. It writes until `Ctrl-C` — do not `kill -9`,
or the pcap will not flush cleanly.

```bash
sudo sidechannel/capture.sh <endpoint-host> run.pcap
```

**2. Generate calls** while the capture runs.

```bash
python -m mdagents_sidechannel.run_experiment --dataset medqa --num_samples 100
```

**3. Slice** the capture against the log. Warns if any call's window held no
packets — usually the capture was not running for the whole call, or filtered
the wrong IP (check `capture.sh`'s resolved address if the endpoint is behind a
CDN).

```bash
python sidechannel/slice_pcap.py \
    --csv runs/logs/smoketest_log.csv --pcap run.pcap \
    --out-csv sliced.csv --out-json sliced_chunks.json
```

`sliced.csv` is one row per call: wire byte and packet counts beside the
application-layer `request_bytes`/`response_bytes`. `sliced_chunks.json` holds
per-call packet detail — chunk sizes, categories, inter-arrival gaps — keyed by
`call_index`.

**4. Enrich** (diagnosis experiments only). Joins each row onto
`data/<dataset>/test.jsonl` by `question_id`, adding `answer`/`answer_idx`. One
question spans several calls, so the answer repeats across rows sharing a
`question_id`.

```bash
python sidechannel/enrich_sliced.py \
    --sliced-csv sliced.csv --dataset medqa --out sliced_enriched.csv
```

**5. Classify.**

```bash
python sidechannel/tier_classifier.py --level call --sliced-csv sliced.csv --log-csv <log>
python sidechannel/diagnosis_classifier.py --level case --sliced-csv sliced_enriched.csv
```

`--level` selects the threat model: `case` sees a whole session and can count
its calls, which makes the tier nearly trivial to recover because MDAgents makes
a near-fixed number of calls per tier; `call` sees one isolated call, with folds
grouped by `question_id` so a case never splits across train and test. Each
script's docstring has the detail.

## Merging runs

`merge_sliced_runs.py` concatenates sliced CSVs from several captures,
renumbering `call_index` and recording `source_run`. `runs/traces/ddxplus_balanced/`
is such a merge of two runs.
