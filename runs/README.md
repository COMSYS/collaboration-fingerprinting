# Traces and results

## Traces — `traces/`

Per-call wire features extracted from the packet captures, which are not
committed. These CSVs are what the classifiers read.

| file | rows | contents |
|---|---|---|
| `medqa/wire-sliced.csv` | 11,852 | wire features per call, MedQA |
| `medqa/call-log.csv` | 11,852 | the paired call log, for `latency_seconds` |
| `ddxplus_balanced/wire-sliced.csv` | 5,531 | wire features per call, 600 balanced DDXPlus cases |
| `ddxplus_balanced/wire-sliced-enriched.csv` | 5,531 | the same, plus the `answer` column the diagnosis classifier needs |

`slice_pcap.py` does not carry timestamps through, so `medqa/wire-sliced.csv`
has none and the MedQA commands pass `--log-csv` as well. The DDXPlus file is a
`merge_sliced_runs.py` output and already carries `latency_seconds`.

The `role` column holds the agent's specialty only. MDAgents' recruiter returns
each role as a title and a description of what that expert should examine in
this case, and the description is clinical narrative about the patient, which
does not belong in a wire trace. It is removed; the title is kept. Both
classifiers bucket every non-canonical role to `other_specialist`, so this
changes no result — verified by diffing their complete output, ablations and
feature importances included, before and after.

## Results — `results/paper/`

| file | claim | value |
|---|---|---|
| `tier-medqa-call.json` | single-call tier recovery, MedQA | **0.744**, floor 0.312 (11,850 calls) |
| `tier-medqa-case.json` | case-level tier recovery, MedQA | **1.000** (1,272 cases) |
| `tier-ddxplus-call.json` | single-call tier recovery, DDXPlus | **0.740**, floor 0.285 (5,531 calls) |
| `tier-ddxplus-case.json` | case-level tier recovery, DDXPlus | **0.994** (600 cases) |
| `diagnosis-ddxplus-call.json` | single-call diagnosis recovery | **0.138** |
| `diagnosis-ddxplus-case.json` | case-level diagnosis recovery | **0.206**, top-3 0.470, top-5 0.648 |

All macro-F1, 5-fold CV, seed 0; 3-class for tier, 12-class for diagnosis. The
tier runs use `StratifiedGroupKFold` grouped by `question_id`.

Earlier runs that no number rests on — a 219-case tier pilot, a 45-diagnosis
run over 220 unbalanced cases, a 12-diagnosis run over the first 250 — are not
kept here.

A fresh run writes to `results/new/` by default, so it never lands among these.
