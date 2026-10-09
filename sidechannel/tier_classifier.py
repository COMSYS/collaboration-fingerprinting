"""Difficulty-tier classifier from wire metadata only.

Question: can an observer who sees only encrypted wire traffic (packet
sizes, counts, timing -- never plaintext) recover which collaboration
tier (basic/intermediate/advanced) MDAgents assigned to a case?

Two threat models, selected with --level:

  case (default): the observer sees an ENTIRE session's calls and can
    count them. sliced.csv's per-call rows are aggregated to one feature
    vector per question_id (a "case") before classifying. In practice
    this makes the tier trivial to recover: MDAgents' own control flow
    makes a near-fixed number of API calls per tier (see main.py's
    process_basic/intermediate/advanced_query), so n_calls alone already
    separates the classes almost perfectly -- a much weaker signal
    (sizes/timing) is never actually needed.

  call: a harder, more realistic model -- the observer sees a single
    isolated call with no session context (no call count, no
    aggregation). Every call inherits its case's tier as its label, and
    is classified independently from its own size/packet/timing features
    alone. Folds are grouped by question_id (StratifiedGroupKFold) so
    calls from the same case never leak across train/test -- otherwise a
    model could partly memorize per-case patterns instead of learning
    genuine per-call signal.

Usage (from the repository root):
    python sidechannel/tier_classifier.py --level case \
        --sliced-csv runs/traces/medqa/wire-sliced.csv \
        --log-csv runs/traces/medqa/call-log.csv

    python sidechannel/tier_classifier.py --level call \
        --sliced-csv runs/traces/medqa/wire-sliced.csv \
        --log-csv runs/traces/medqa/call-log.csv

--log-csv is optional and only used to bring in call latency (sliced.csv
itself has no timestamp columns -- slice_pcap.py never carries
call_start_ts/call_end_ts into its output, only the paired raw log CSV
has them). If omitted, timing features are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--sliced-csv', required=True)
parser.add_argument('--log-csv', default=None, help='raw call log CSV, for latency_seconds; optional')
parser.add_argument('--level', choices=['case', 'call'], default='case')
parser.add_argument('--n-splits', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out-dir', default='runs/results/new',
                     help='where to save results JSON; see --no-save to disable')
parser.add_argument('--no-save', action='store_true', help='skip writing results to disk')
args = parser.parse_args()


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


def save_results(results):
    if args.no_save:
        return
    run_tag = Path(args.sliced_csv).parent.name or Path(args.sliced_csv).stem
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    out_path = Path(args.out_dir) / f"{args.level}_{run_tag}_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=to_jsonable)
    print(f"Results saved to: {out_path}")

CANONICAL_ROLES = ['medical expert', 'recruiter', 'medical assistant', 'Moderator', 'decision maker']


def bucket_role(role):
    return role if role in CANONICAL_ROLES else 'other_specialist'


def make_models(seed):
    lr = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed)),
    ])
    gbm = LGBMClassifier(objective='multiclass', class_weight='balanced',
                          n_estimators=200, random_state=seed, verbosity=-1)
    return {'LogisticRegression': lr, 'LightGBM': gbm}


def load_base():
    df = pd.read_csv(args.sliced_csv, dtype={'question_id': str})
    df['assigned_tier'] = df['assigned_tier'].fillna('')

    print(f"case count (unique question_id): {df['question_id'].nunique()}")
    print()
    print("literal groupby(question_id).assigned_tier.first().value_counts() "
          "(requested diagnostic):")
    print(df.groupby('question_id')['assigned_tier'].first().value_counts())
    print()
    print("NOTE: that is 100% blank. Every case's first logged call is the "
          "untagged difficulty-determination call (see main.py:34-38 -- "
          "set_log_context(tier=None) happens before determine_difficulty "
          "runs, and only gets set afterward), so .first() never reaches a "
          "real tier. Using first NON-BLANK tier per case instead:")

    def first_nonblank(s):
        nonblank = s[s != '']
        return nonblank.iloc[0] if len(nonblank) else np.nan

    case_tier = df.groupby('question_id')['assigned_tier'].apply(first_nonblank)
    print(case_tier.value_counts(dropna=False))
    print()

    n_no_tier = case_tier.isna().sum()
    if n_no_tier:
        print(f"Dropping {n_no_tier} case(s) with no tagged tier at all "
              f"(likely an interrupted/crashed question): "
              f"{list(case_tier[case_tier.isna()].index)}")
    case_tier = case_tier.dropna()
    print()

    if 'latency_seconds' in df.columns:
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
            print("--log-csv not given: sliced.csv has no timestamp columns "
                  "(slice_pcap.py doesn't carry call_start_ts/call_end_ts into "
                  "its output), so timing features are NOT present.")
    print()

    df['handshake_bytes'] = df['request_bytes_wire_handshake'] + df['response_bytes_wire_handshake']
    df['role_bucket'] = df['role'].map(bucket_role)

    # drop rows belonging to cases with no tagged tier at all
    df = df[df['question_id'].isin(case_tier.index)].copy()

    return df, case_tier, have_timing


def run_ablation_report(full_feat, y, ablations, cv, groups=None):
    print("=" * 70)
    print(f"ABLATION: mean macro-F1 over {args.n_splits}-fold stratified CV")
    print("=" * 70)
    results = []
    for name, cols in ablations:
        if '[SKIPPED' in name:
            print(f"{name}: skipped")
            continue
        X = full_feat[cols].values
        row = {'feature_set': name, 'n_features': len(cols)}
        for model_name, model in make_models(args.seed).items():
            scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='f1_macro')
            row[model_name] = f"{scores.mean():.3f} +/- {scores.std():.3f}"
        results.append(row)
        print(f"{name} ({len(cols)} features): "
              + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ('feature_set', 'n_features')))
    print()
    return pd.DataFrame(results)


def run_primary_report(X_obs, y, feature_names, cv, groups=None, label='PRIMARY RESULT'):
    print("=" * 70)
    print(f"{label}: LightGBM on full observable feature set")
    print("(role excluded -- label-adjacent, see the '+ role' ablation for that upper bound)")
    print("=" * 70)
    gbm = LGBMClassifier(objective='multiclass', class_weight='balanced',
                          n_estimators=200, random_state=args.seed, verbosity=-1)
    y_pred = cross_val_predict(gbm, X_obs, y, groups=groups, cv=cv)

    macro_f1 = f1_score(y, y_pred, average='macro')
    report_dict = classification_report(y, y_pred, digits=3, output_dict=True)
    print(f"macro-F1 (OOF): {macro_f1:.3f}")
    print()
    print(classification_report(y, y_pred, digits=3))
    print("confusion matrix (rows=true, cols=predicted):")
    labels = sorted(pd.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))
    print()

    gbm_full = LGBMClassifier(objective='multiclass', class_weight='balanced',
                               n_estimators=200, random_state=args.seed, verbosity=-1)
    gbm_full.fit(X_obs, y)
    importances = pd.Series(gbm_full.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("=" * 70)
    print("LightGBM feature importances (fit on all data)")
    print("=" * 70)
    print(importances.to_string())
    print()

    primary_info = {
        'macro_f1_oof': macro_f1,
        'classification_report': report_dict,
        'confusion_matrix': {'labels': list(labels), 'matrix': cm},
        'feature_importances': importances,
    }
    return y_pred, importances, primary_info


def run_case_level(df, case_tier, have_timing):
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

    role_counts = g['role_bucket'].value_counts().unstack(fill_value=0)
    role_counts.columns = [f'role_n_{c.replace(" ", "_")}' for c in role_counts.columns]

    feat = feat.loc[case_tier.index]
    role_counts = role_counts.loc[case_tier.index]
    y = case_tier

    print(f"Feature table: {feat.shape[0]} cases x {feat.shape[1]} base features "
          f"(+{role_counts.shape[1]} role-bucket features for the label-adjacent ablation)")
    print()

    set_a = ['n_calls']
    set_b = set_a + [c for c in feat.columns if 'wire_appdata' in c]
    set_c = set_b + (['total_latency_s', 'max_latency_s'] if have_timing else [])
    set_d = set_c + ['n_connection_opens']
    full_observable = set_d + [c for c in feat.columns if c not in set_d]
    ablations = [
        ('(a) n_calls only', set_a),
        ('(b) + appdata sizes', set_b),
        ('(c) + timing' + ('' if have_timing else ' [SKIPPED: no --log-csv]'), set_c),
        ('(d) + connection-opens (= full observable set)', full_observable),
        ('(e) + role [label-adjacent upper bound]', full_observable + list(role_counts.columns)),
    ]

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    full_feat = pd.concat([feat, role_counts], axis=1)

    ablation_df = run_ablation_report(full_feat, y, ablations, cv)

    X_obs = full_feat[full_observable].values
    _, _, primary_info = run_primary_report(
        X_obs, y, full_observable, cv,
        label='PRIMARY RESULT (case-level, sees the whole session)')

    n_advanced = int((y == 'advanced').sum())
    caveat = None
    if n_advanced < 2 * args.n_splits:
        caveat = (f"only {n_advanced} 'advanced' cases exist -- its precision/recall are "
                  f"estimated from a handful of examples and should be treated as noisy, "
                  f"not a reliable estimate.")
        print(f"CAVEAT: {caveat}")

    return {
        'case_tier_distribution': y.value_counts().to_dict(),
        'n_cases': len(y),
        'ablation': ablation_df,
        'primary': primary_info,
        'caveat_advanced_rarity': caveat,
    }


def run_call_level(df, case_tier, have_timing):
    df = df.copy()
    original_tier = df['assigned_tier']  # per-row tag before inheritance, '' = untagged
    df['tier'] = df['question_id'].map(case_tier)
    df['connection_open'] = (df['handshake_bytes'] > 0).astype(int)

    feat = df[['question_id', 'tier',
               'request_bytes_wire_appdata', 'response_bytes_wire_appdata',
               'n_up_packets', 'n_down_packets', 'connection_open', 'role_bucket']].copy()
    if have_timing:
        feat['latency_s'] = df['latency_seconds']

    role_dummies = pd.get_dummies(feat['role_bucket'], prefix='role')

    y = feat['tier']
    groups = feat['question_id']

    # ---- Labeling scaffolding -------------------------------------------
    print("=" * 70)
    print("LABELING METHOD")
    print("=" * 70)
    n_own_tag = int((original_tier[df.index] != '').sum())
    n_inherited = int((original_tier[df.index] == '').sum())
    print("Every call inherits its case's final non-blank tier via "
          "question_id (df['tier'] = df['question_id'].map(case_tier)). "
          "Blank-tier calls (the difficulty-determination call and any other "
          "untagged pre-escalation calls) are NOT dropped -- they're kept "
          "and labeled with their case's tier, since they're real traffic "
          "an observer would also see. Only calls belonging to a case with "
          "NO tagged tier anywhere (case 627) are excluded entirely.")
    print(f"  calls with their own row already tagged:            {n_own_tag}")
    print(f"  calls with a blank row tag, tier inherited from case: {n_inherited}")
    print(f"  total call-level rows used:                          {len(feat)} "
          f"(across {groups.nunique()} cases)")
    print(f"Resulting per-class call counts: {y.value_counts().to_dict()}")
    print()

    set_a = ['request_bytes_wire_appdata', 'response_bytes_wire_appdata']
    set_b = set_a + ['n_up_packets', 'n_down_packets']
    set_c = set_b + (['latency_s'] if have_timing else [])
    set_d = set_c + ['connection_open']
    ablations = [
        ('(a) appdata sizes only (single call)', set_a),
        ('(b) + packet counts', set_b),
        ('(c) + timing' + ('' if have_timing else ' [SKIPPED: no --log-csv]'), set_c),
        ('(d) + connection-open flag (= full observable set)', set_d),
        ('(e) + role [label-adjacent upper bound]', set_d + list(role_dummies.columns)),
    ]

    full_feat = pd.concat([feat, role_dummies], axis=1)

    # ---- Materialize folds ONCE -- reused for every ablation entry, the ----
    # ---- majority baseline, and the primary model (fair, identical splits,
    # ---- and each model is trained/scored exactly once per feature set).
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(full_feat, y, groups))

    print("=" * 70)
    print(f"PER-FOLD 'advanced' SUPPORT (test-fold sizes, {args.n_splits} folds)")
    print("=" * 70)
    per_fold_support = []
    for i, (_, test_idx) in enumerate(folds):
        counts = y.iloc[test_idx].value_counts()
        fold_row = {'fold': i, 'total': len(test_idx),
                    'basic': int(counts.get('basic', 0)),
                    'intermediate': int(counts.get('intermediate', 0)),
                    'advanced': int(counts.get('advanced', 0))}
        per_fold_support.append(fold_row)
        print(f"  fold {i}: total={fold_row['total']:5d}  "
              f"basic={fold_row['basic']:5d}  "
              f"intermediate={fold_row['intermediate']:4d}  "
              f"advanced={fold_row['advanced']:3d}")
    print()

    # ---- Majority-class baseline, same folds as everything else ---------
    X_d = full_feat[set_d].values
    dummy_scores = cross_val_score(DummyClassifier(strategy='most_frequent'), X_d, y, cv=folds, scoring='f1_macro')
    print("=" * 70)
    print("MAJORITY-CLASS BASELINE (always predict 'basic'), same folds")
    print("=" * 70)
    print(f"macro-F1: {dummy_scores.mean():.3f} +/- {dummy_scores.std():.3f}  (floor for the 0.744 result)")
    print()

    ablation_df = run_ablation_report(full_feat, y, ablations, folds)

    X_obs = full_feat[set_d].values
    y_pred, importances, primary_info = run_primary_report(
        X_obs, y, set_d, folds,
        label='PRIMARY RESULT (call-level, single isolated call, no session context)')

    # ---- Boundary breakdowns: macro-F1 over 3 classes of very different --
    # ---- size (10440/1356/54) hides where the signal actually is/isn't.
    print("=" * 70)
    print("BOUNDARY BREAKDOWN")
    print("=" * 70)
    y_true_bin = np.where(y == 'basic', 'basic', 'rest')
    y_pred_bin = np.where(y_pred == 'basic', 'basic', 'rest')
    basic_vs_rest_f1 = f1_score(y_true_bin, y_pred_bin, pos_label='basic')
    print(f"basic-vs-rest F1 (positive class = 'basic', full population, n={len(y)}): "
          f"{basic_vs_rest_f1:.3f}")
    print()

    mask = y.isin(['intermediate', 'advanced'])
    print(f"intermediate-vs-advanced separation (conditional on true label being one of "
          f"these two, n={mask.sum()}; predictions that fell through to 'basic' count as "
          f"wrong for both, not excluded):")
    cm_ia = confusion_matrix(y[mask], y_pred[mask], labels=['intermediate', 'advanced', 'basic'])
    cm_ia_df = pd.DataFrame(cm_ia, index=['true_intermediate', 'true_advanced', 'true_basic (n/a, mask excludes it)'],
                             columns=['pred_intermediate', 'pred_advanced', 'pred_basic (leaked)'])
    print(cm_ia_df.iloc[:2])  # 3rd row is all-zero by construction (mask excludes true='basic')
    ia_f1 = f1_score(y[mask], y_pred[mask], labels=['intermediate', 'advanced'], average='macro')
    print(f"macro-F1 (intermediate vs advanced only, 'leaked to basic' counted as wrong): {ia_f1:.3f}")
    print()

    # ---- Timing confirmation ---------------------------------------------
    print("=" * 70)
    print("TIMING CONFIRMATION")
    print("=" * 70)
    print("Ablation (b)->(c) macro-F1 jump quantifies whether timing adds signal beyond size/packets "
          "(see the ABLATION table above for the exact (b) and (c) numbers).")
    print("Top-5 feature importances (from the already-fit LightGBM model above):")
    print(importances.head(5).to_string())
    print()

    print("CAVEAT: folds are grouped by question_id (StratifiedGroupKFold) so calls from "
          "the same case never split across train/test -- without that, a model could "
          "partly memorize per-case patterns rather than learning genuine per-call signal.")

    return {
        'labeling': {
            'n_own_tag': n_own_tag,
            'n_inherited': n_inherited,
            'per_class_call_counts': y.value_counts().to_dict(),
            'n_calls': len(feat),
            'n_cases': int(groups.nunique()),
        },
        'per_fold_advanced_support': per_fold_support,
        'majority_baseline': {
            'strategy': 'most_frequent (always basic)',
            'macro_f1_mean': float(dummy_scores.mean()),
            'macro_f1_std': float(dummy_scores.std()),
        },
        'ablation': ablation_df,
        'primary': primary_info,
        'boundary_breakdown': {
            'basic_vs_rest_f1': float(basic_vs_rest_f1),
            'intermediate_vs_advanced': {
                'n': int(mask.sum()),
                'confusion_matrix': cm_ia_df.iloc[:2],
                'macro_f1': float(ia_f1),
            },
        },
    }


def main():
    df, case_tier, have_timing = load_base()
    if args.level == 'case':
        results = run_case_level(df, case_tier, have_timing)
    else:
        results = run_call_level(df, case_tier, have_timing)

    results['meta'] = {
        'level': args.level,
        'sliced_csv': args.sliced_csv,
        'log_csv': args.log_csv,
        'n_splits': args.n_splits,
        'seed': args.seed,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    save_results(results)


if __name__ == '__main__':
    main()
