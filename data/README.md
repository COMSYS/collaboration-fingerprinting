# Datasets

The splits are **not** included. No number depends on them: the results are
computed from the traces in `../runs/traces/`, which are committed, as is the
enriched trace the diagnosis classifier needs. Splits are required only to
collect new traces.

Rebuild into this directory as `data/<dataset>/{train,test}.jsonl`.

## DDXPlus

CC-BY ([Fansi Tchango et al., NeurIPS 2022](https://arxiv.org/abs/2205.09148)).
Download the raw release, then:

```bash
python scripts/prepare_ddxplus.py --raw-dir <raw-ddxplus-dir> --out-dir data/ddxplus
python scripts/sample_ddxplus_balanced.py --n-classes 12 --per-class 50 --seed 42
```

Both are seeded and the balanced sampler walks classes in a fixed rank order, so
this reproduces our splits exactly.

## MedQA

From [jind11/MedQA](https://github.com/jind11/MedQA) (MIT), the English `US`
4-option split. `test.jsonl` is their test split unchanged, 1,273 questions.

`train.jsonl` is **not** their training split. It is a 19-row pool MDAgents
draws five few-shot examplers from per call. Using the full 10,178-row split
changes which examplers appear, which changes prompt sizes, which changes the
traffic an adversary sees.

## Verifying a rebuild

```bash
shasum -a 256 data/*/*.jsonl
```

| file | rows | sha256 |
|---|---|---|
| `ddxplus/train.jsonl` | 200 | `62c52674a689c420c180777be31af62a1e3418c54c4a330193ccce65afa8030b` |
| `ddxplus/test.jsonl` | 300 | `cb7cb89210cad714d53984e4b08c0838ca7a48157c5f50ee68428cf866fcbd87` |
| `ddxplus_balanced/train.jsonl` | 200 | `62c52674a689c420c180777be31af62a1e3418c54c4a330193ccce65afa8030b` |
| `ddxplus_balanced/test.jsonl` | 600 | `02a71e7a8542ef4b4ed8ca76f1bdcb15b7725dc96fc77c11e8594873612315cd` |
| `medqa/train.jsonl` | 19 | `b83541b211647aa26ed50f4b4708b05bdedb5e23f0d14d571e0a8fe729b1a754` |
| `medqa/test.jsonl` | 1273 | `cfa75cbcc53c29b023752b0a395ff49d8202121041392735ae44771a5ec734a3` |

The two `train.jsonl` files are identical: the balanced sampler rebuilds only
the test split and reuses the exampler pool.
