"""Diagnosis-prediction experiment on DDXPlus wire metadata.

Same question as tier_classifier.py -- can an observer who sees only
encrypted wire traffic recover something they shouldn't -- but a harder
target: not which of 3 difficulty tiers was assigned, but which of ~45
diagnoses (DDXPlus's `answer` column, joined in by enrich_sliced.py
--dataset ddxplus) the case was actually about.

Reuses tier_classifier.py's pipeline (same feature engineering, same two
adversaries, same StratifiedGroupKFold-by-question_id discipline) with the
label swapped from assigned_tier to answer. Kept as a separate script
rather than importing from tier_classifier.py because that module builds
its argparse `args` at import time, which would collide with this
script's own CLI arguments.

Two adversaries (both run by default, see --level):
  case: sees a whole session, features aggregated per question_id.
  call: sees one isolated call, no session context, no call count.

Usage (from the repository root):
    python sidechannel/diagnosis_classifier.py \
        --sliced-csv runs/captures/sliced_enriched.csv \
        --log-csv runs/logs/smoketest_log.csv

--sliced-csv must be enrich_sliced.py's output (needs an `answer` column)
-- run `python sidechannel/enrich_sliced.py --dataset ddxplus ...` first if
you only have the plain sliced.csv.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--sliced-csv', required=True, help='enrich_sliced.py output -- needs an `answer` column')
parser.add_argument('--log-csv', default=None, help='raw call log CSV, for latency_seconds; optional')
parser.add_argument('--level', choices=['case', 'call', 'both'], default='both')
parser.add_argument('--n-splits', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out-dir', default='runs/results/new',
                     help='where to save results JSON; see --no-save to disable')
parser.add_argument('--no-save', action='store_true', help='skip writing results to disk')
args = parser.parse_args()

CANONICAL_ROLES = ['medical expert', 'recruiter', 'medical assistant', 'Moderator', 'decision maker']


def bucket_role(role):
    return role if role in CANONICAL_ROLES else 'other_specialist'


def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    raise TypeError(f"not JSON serializable: {type(obj)} ({obj!r})")


def save_results(results, level):
    if args.no_save:
        return
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = Path(args.out_dir) / f"{level}_ddxplus_diagnosis_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=to_jsonable)
    print(f"Results saved to: {out_path}")


def make_models(seed):
    lr = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed)),
    ])
    # min_child_samples lowered from LightGBM's default (20) -- with ~220
    # cases across ~45 classes (average ~5 cases/class), the default would
    # forbid most splits outright on a dataset this small.
    gbm = LGBMClassifier(objective='multiclass', class_weight='balanced',
                          n_estimators=200, min_child_samples=3, random_state=seed, verbosity=-1)
    return {'LogisticRegression': lr, 'LightGBM': gbm}


def load_base():
    df = pd.read_csv(args.sliced_csv, dtype={'question_id': str})
    if 'answer' not in df.columns:
        raise SystemExit(f"{args.sliced_csv} has no 'answer' column -- run "
                          f"sidechannel/enrich_sliced.py --dataset ddxplus on it first")
    df['assigned_tier'] = df['assigned_tier'].fillna('')

    # ---- Step 1: diagnosis distribution --------------------------------
    case_answer = df.groupby('question_id')['answer'].first()
    print("=" * 70)
    print("DIAGNOSIS DISTRIBUTION (groupby(question_id).answer.first().value_counts())")
    print("=" * 70)
    print(f"case count: {len(case_answer)}")
    vc = case_answer.value_counts()
    print(f"distinct diagnoses: {len(vc)}")
    print(vc.to_string())
    n_rare = int((vc < args.n_splits).sum())
    n_singleton = int((vc == 1).sum())
    print()
    print(f"diagnoses with fewer than {args.n_splits} cases: {n_rare} "
          f"(their per-class recall under {args.n_splits}-fold CV will be unstable)")
    print(f"diagnoses with exactly 1 case: {n_singleton} "
          f"(these can NEVER be in both train and test in the same fold -- "
          f"a model can never learn them, recall is 0 by construction, not a modeling failure)")
    print()

    # ---- Step 2: confirm answer is constant within question_id ---------
    n_unique_per_case = df.groupby('question_id')['answer'].nunique()
    inconsistent = n_unique_per_case[n_unique_per_case > 1]
    if len(inconsistent):
        print(f"WARNING: {len(inconsistent)} case(s) have an inconsistent 'answer' across their "
              f"own rows (should never happen -- enrichment joins one dataset row per case): "
              f"{list(inconsistent.index)}")
    else:
        print("Confirmed: 'answer' is constant within every question_id (no inconsistent cases).")
    print()

    # ---- Step 3: majority-diagnosis floor + chance ----------------------
    n_classes = case_answer.nunique()
    chance = 1.0 / n_classes
    print(f"chance level (1 / n_classes): {chance:.4f}  (n_classes={n_classes}, not hardcoded)")
    print("(majority-diagnosis macro-F1 floor is reported per-adversary below, on that "
          "adversary's own CV folds, alongside its primary result)")
    print()

    if 'latency_seconds' in df.columns:
        # sidechannel/merge_sliced_runs.py already baked this in -- call_index
        # resets to 1 in every fresh process, so re-joining a --log-csv here
        # (which assumes call_index is unique) would corrupt a merged multi-run
        # file. See merge_sliced_runs.py's docstring for why.
        have_timing = True
        print("'latency_seconds' already present (this looks like a merge_sliced_runs.py "
              "output) -- using it directly, ignoring --log-csv if given.")
    else:
        have_timing = args.log_csv is not None
        if have_timing:
            log_df = pd.read_csv(args.log_csv, usecols=['call_index', 'latency_seconds'])
            df = df.merge(log_df, on='call_index', how='left')
            print(f"Joined latency_seconds from {args.log_csv} via call_index.")
        else:
            print("--log-csv not given: timing features skipped.")
    print()

    df['handshake_bytes'] = df['request_bytes_wire_handshake'] + df['response_bytes_wire_handshake']
    df['role_bucket'] = df['role'].map(bucket_role)

    return df, case_answer, have_timing, n_classes


def run_multiclass_report(X_obs, y, feature_names, folds, all_classes, label):
    print("=" * 70)
    print(f"{label}")
    print("=" * 70)

    # majority-diagnosis floor, same folds as the primary model
    dummy_scores = cross_val_score(DummyClassifier(strategy='most_frequent'), X_obs, y,
                                    cv=folds, scoring='f1_macro')
    print(f"majority-diagnosis floor (macro-F1, same folds): "
          f"{dummy_scores.mean():.3f} +/- {dummy_scores.std():.3f}")
    print(f"chance level (1/{len(all_classes)}): {1.0 / len(all_classes):.4f}")
    print()

    gbm = LGBMClassifier(objective='multiclass', class_weight='balanced',
                          n_estimators=200, min_child_samples=3, random_state=args.seed, verbosity=-1)
    y_proba = cross_val_predict(gbm, X_obs, y, cv=folds, method='predict_proba')
    # cross_val_predict aligns predict_proba columns to sorted(unique(y)) globally,
    # zero-padding any class a given fold's training split never saw -- same order
    # as all_classes below, so argmax reproduces plain predict() without refitting.
    y_pred = all_classes[np.argmax(y_proba, axis=1)]

    macro_f1 = f1_score(y, y_pred, average='macro')
    report_dict = classification_report(y, y_pred, digits=3, output_dict=True, zero_division=0)
    print(f"macro-F1 (OOF): {macro_f1:.3f}")
    print()
    print(classification_report(y, y_pred, digits=3, zero_division=0))

    top3 = top_k_accuracy_score(y, y_proba, k=3, labels=all_classes)
    top5 = top_k_accuracy_score(y, y_proba, k=5, labels=all_classes)
    print(f"top-3 accuracy (true diagnosis among top 3 predictions): {top3:.3f}")
    print(f"top-5 accuracy (true diagnosis among top 5 predictions): {top5:.3f}")
    print()

    cm = confusion_matrix(y, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in all_classes], columns=[f"pred_{l}" for l in all_classes])
    print("confusion matrix (rows=true, cols=predicted; large -- see saved JSON for the full matrix):")
    print(cm_df.iloc[:8, :8])
    print("... (truncated for console; full matrix is in the saved JSON)")
    print()

    gbm_full = LGBMClassifier(objective='multiclass', class_weight='balanced',
                               n_estimators=200, min_child_samples=3, random_state=args.seed, verbosity=-1)
    gbm_full.fit(X_obs, y)
    importances = pd.Series(gbm_full.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("LightGBM feature importances (fit on all data):")
    print(importances.to_string())
    print()

    return {
        'majority_floor': {'macro_f1_mean': float(dummy_scores.mean()), 'macro_f1_std': float(dummy_scores.std())},
        'chance_level': 1.0 / len(all_classes),
        'n_classes': int(len(all_classes)),
        'macro_f1_oof': float(macro_f1),
        'classification_report': report_dict,
        'top_3_accuracy': float(top3),
        'top_5_accuracy': float(top5),
        'confusion_matrix': {'labels': list(all_classes), 'matrix': cm},
        'feature_importances': importances,
    }


def run_case_level(df, case_answer, have_timing, n_classes):
    g = df.groupby('question_id')
    feat = pd.DataFrame(index=g.size().index)
    feat['n_calls'] = g.size()

    for col in ['request_bytes_wire_appdata', 'response_bytes_wire_appdata']:
        feat[f'{col}_sum'] = g[col].sum()
        feat[f'{col}_max'] = g[col].max()
        feat[f'{col}_mean'] = g[col].mean()

    for col in ['n_up_packets', 'n_down_packets']:
        feat[f'{col}_sum'] = g[col].sum()
        feat[f'{col}_max'] = g[col].max()

    feat['n_connection_opens'] = g.apply(lambda d: (d['handshake_bytes'] > 0).sum())

    if have_timing:
        feat['total_latency_s'] = g['latency_seconds'].sum()
        feat['max_latency_s'] = g['latency_seconds'].max()

    feat = feat.loc[case_answer.index]
    y = case_answer
    all_classes = np.sort(y.unique())

    print(f"Case-level feature table: {feat.shape[0]} cases x {feat.shape[1]} observable features "
          f"(role excluded -- not part of the tier run's observable set either)")
    print()

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    X_index = np.arange(len(y))
    folds = list(cv.split(X_index, y))

    result = run_multiclass_report(feat.values, y, list(feat.columns), folds, all_classes,
                                    label='SESSION-LEVEL ADVERSARY (case aggregates)')
    result['level'] = 'case'
    result['n_cases'] = len(y)
    return result


def run_call_level(df, case_answer, have_timing):
    df = df.copy()
    df['answer'] = df['question_id'].map(case_answer)
    df['connection_open'] = (df['handshake_bytes'] > 0).astype(int)

    feat = df[['question_id', 'answer',
               'request_bytes_wire_appdata', 'response_bytes_wire_appdata',
               'n_up_packets', 'n_down_packets', 'connection_open']].copy()
    if have_timing:
        feat['latency_s'] = df['latency_seconds']

    y = feat['answer']
    groups = feat['question_id']
    all_classes = np.sort(y.unique())

    feature_cols = ['request_bytes_wire_appdata', 'response_bytes_wire_appdata',
                     'n_up_packets', 'n_down_packets', 'connection_open']
    if have_timing:
        feature_cols.append('latency_s')

    print(f"Call-level feature table: {len(feat)} individual calls (no aggregation, no n_calls) "
          f"across {groups.nunique()} cases, {len(feature_cols)} observable features")
    print()

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(feat[feature_cols].values, y, groups))

    result = run_multiclass_report(feat[feature_cols].values, y, feature_cols, folds, all_classes,
                                    label='SINGLE-CALL ADVERSARY (no session context)')
    result['level'] = 'call'
    result['n_calls'] = len(feat)
    result['n_cases'] = int(groups.nunique())
    return result


def main():
    df, case_answer, have_timing, n_classes = load_base()

    if args.level in ('case', 'both'):
        case_result = run_case_level(df, case_answer, have_timing, n_classes)
        case_result['meta'] = {'sliced_csv': args.sliced_csv, 'log_csv': args.log_csv,
                                'n_splits': args.n_splits, 'seed': args.seed,
                                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
        save_results(case_result, 'case')

    if args.level in ('call', 'both'):
        call_result = run_call_level(df, case_answer, have_timing)
        call_result['meta'] = {'sliced_csv': args.sliced_csv, 'log_csv': args.log_csv,
                                'n_splits': args.n_splits, 'seed': args.seed,
                                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
        save_results(call_result, 'call')


if __name__ == '__main__':
    main()
