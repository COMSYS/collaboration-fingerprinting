# Collaboration Fingerprinting: Side-Channel Leakage in Adaptive Multi-Agent Systems

Code and traces for our study of what encrypted inter-agent traffic reveals
about an adaptive medical multi-agent system. An observer who sees only packet
sizes, counts and timing — never plaintext — recovers the collaboration tier
MDAgents assigns to a case, and narrows the diagnosis.

## Relationship to MDAgents

This repository contains **only our own code**. The system under study,
[MDAgents](https://github.com/mitmedialab/MDAgents) (Kim et al., NeurIPS 2024),
is third-party work and is not redistributed here. `setup_mdagents.sh` clones it
at a pinned commit, and `mdagents_sidechannel/upstream.py` loads, corrects and
instruments it in memory at import time; nothing is written back to the clone.

```
mdagents_sidechannel/   instrumentation and experiment driver
sidechannel/            pcap slicing and the two adversary classifiers
scripts/                dataset preparation
runs/traces/            the traces every number is computed from
runs/results/paper/     the six results backing the paper
data/                   third-party splits, not included — see data/README.md
```

`runs/README.md` maps each trace and each stored result to the claim it backs.

## Setup

To **reproduce the published numbers** this is the whole of it. The classifiers
read the committed traces, so there is no MDAgents clone, no API key and no
endpoint involved:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer. On macOS `lightgbm` also needs `brew install libomp`.

`requirements.txt` pins scikit-learn and lightgbm. Those versions are part of
the result: on current releases the MedQA single-call figure comes out as 0.753
rather than 0.744.

To **collect new traces** you additionally need upstream MDAgents, which raises
the floor to **Python 3.12**: its `utils.py` has a backslash inside an f-string
expression, a `SyntaxError` before 3.12 and legal from 3.12 on
([PEP 701](https://peps.python.org/pep-0701/)). A syntax error cannot be patched
at runtime, so we require the interpreter that parses it.

```bash
pip install -r requirements-collect.txt
./setup_mdagents.sh      # clones MDAgents, pins 3adbd76, checks our corrections apply
```

## Reproducing the paper's numbers

Computed from the committed traces, not from a fresh run: LLM calls are not
deterministic, so collecting again produces new traffic rather than the same
traffic.

```bash
# tier recovery, MedQA — 0.744 single-call, 1.000 case-level, floor 0.312
python sidechannel/tier_classifier.py --level call \
    --sliced-csv runs/traces/medqa/wire-sliced.csv \
    --log-csv runs/traces/medqa/call-log.csv

# tier recovery, DDXPlus — 0.740 single-call, 0.994 case-level, floor 0.285
python sidechannel/tier_classifier.py --level call \
    --sliced-csv runs/traces/ddxplus_balanced/wire-sliced.csv

# diagnosis recovery, DDXPlus — 0.206 case-level, 0.138 single-call, 0.083 chance
python sidechannel/diagnosis_classifier.py --level case \
    --sliced-csv runs/traces/ddxplus_balanced/wire-sliced-enriched.csv
```

Packet captures are not included; `runs/traces/` holds the per-call wire
features derived from them. Figure generation is not part of this repository.

## Collecting new traces

```bash
export MDA_BASE_URL=https://<your-endpoint>/v1
export LLM_API_KEY=...
```

`sidechannel/README.md` has the pipeline: capture, run, slice, optionally
enrich, classify. `run_experiment.py --start_index` resumes an interrupted run.

### Backend

No endpoint is hardcoded. Upstream pins model names in several places, so our
client overrides the `model` field on every request and one backend serves every
call.

| variable | default | meaning |
|---|---|---|
| `MDA_MODEL` | `gpt-oss-120b` | model sent on the wire |
| `MDA_BASE_URL` | *(required)* | OpenAI-compatible endpoint serving `MDA_MODEL` |
| `MDA_API_KEY_ENV` | `LLM_API_KEY` | name of the variable holding the key |
| `MDA_STRIP_TEMPERATURE` | `1` | drop `temperature`, matching our traces |

This means `determine_difficulty` runs on the configured backend where stock
MDAgents would use `gpt-3.5`. Our published runs behaved the same way; it is a
deliberate deviation, not upstream behaviour.

## Corrections applied to upstream

Three defects abort an MDAgents run partway through, so all three had to be
corrected to collect our traces. `upstream.py` applies them in memory and
**fails loudly** if upstream no longer matches an anchor, rather than running
without a fix and producing different data.

| defect | correction |
|---|---|
| recruiter output not matching `N. role - description` raises `IndexError` | skip the malformed entry |
| an expert can name an agent number that does not exist | drop out-of-range indices |
| a recruited MDT with no parseable members yields an unusable empty `Group` | skip the group with a warning |

A fourth — the f-string syntax error — is handled by the Python requirement.

## Dataset names change the traffic

Upstream branches on the literal string `'medqa'`: for that name it appends
answer options to the question and generates five few-shot examplers, each an
extra API call. Our runs did not all take that path, and the traces record it.

| dataset | upstream path | calls per basic case |
|---|---|---|
| `medqa` | `'medqa'` — options and examplers | 9 |
| `ddxplus_balanced` | fallback — neither | 4 |

`run_experiment.py` reproduces this mapping by default (`UPSTREAM_ALIAS`);
`--upstream-dataset` overrides it. Running DDXPlus down the `'medqa'` path would
change both the prompts and the traffic.

## Citing

```bibtex
@inproceedings{2026_flueh_collaboration-fingerprinting,
  author = {Flüh, Marlena AND Wehrle, Klaus AND Pennekamp, Jan},
  title = {{Poster: Collaboration Fingerprinting Reveals What Adaptive Multi-Agent LLM Systems Process on the Wire}},
  booktitle = {{Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security}},
  year = {2026},
  doi = {10.1145/3830454.3846427},
  url = {https://www.comsys.rwth-aachen.de/publication/2026/2026_flueh_collaboration-fingerprinting/2026_flueh_collaboration-fingerprinting.pdf},
  publisher = {{ACM}},
  isbn = {{979-8-4007-2871-6}},
}

@inproceedings{kim2024mdagents,
  title     = {MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making},
  author    = {Kim, Yubin and Park, Chanwoo and Jeong, Hyewon and others},
  booktitle = {NeurIPS},
  year      = {2024}
}
```
