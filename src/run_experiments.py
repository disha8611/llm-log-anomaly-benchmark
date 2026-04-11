"""
Paper 1 - Run traditional ML experiments WITHOUT matplotlib.
Generates results JSON that can be plotted later.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEEDS = [42, 123, 456, 789, 1024]


# ============================================================
# SIMPLIFIED DRAIN PARSER
# ============================================================

class SimpleDrainParser:
    def __init__(self, sim_th=0.4):
        self.sim_th = sim_th
        self.clusters = []

    def _similarity(self, seq1, seq2):
        if len(seq1) != len(seq2):
            return 0.0
        return sum(1 for a, b in zip(seq1, seq2) if a == b) / len(seq1)

    def parse(self, messages):
        template_ids = []
        for msg in messages:
            tokens = str(msg).strip().split()
            matched = False
            for i, cluster in enumerate(self.clusters):
                if len(cluster) == len(tokens):
                    if self._similarity(tokens, cluster) >= self.sim_th:
                        # Update cluster template
                        self.clusters[i] = [
                            t1 if t1 == t2 else "<*>"
                            for t1, t2 in zip(tokens, cluster)
                        ]
                        template_ids.append(i)
                        matched = True
                        break
            if not matched:
                self.clusters.append(tokens)
                template_ids.append(len(self.clusters) - 1)
        return template_ids


# ============================================================
# SYNTHETIC DATA GENERATOR (for testing pipeline)
# ============================================================

def generate_synthetic_hdfs_data(n_sessions=5000, n_templates=50):
    """Generate synthetic HDFS-like log data for pipeline testing."""
    print(f"[INFO] Generating synthetic HDFS data: {n_sessions} sessions, {n_templates} templates")
    np.random.seed(42)

    anomaly_rate = 0.03  # 3% anomalous (matches real HDFS)
    n_anomalous = int(n_sessions * anomaly_rate)

    X = np.zeros((n_sessions, n_templates))

    # Normal sessions: follow typical patterns
    for i in range(n_sessions - n_anomalous):
        n_events = np.random.randint(5, 30)
        # Normal sessions have predictable template distributions
        common_templates = np.random.choice(n_templates // 3, n_events, replace=True)
        for t in common_templates:
            X[i][t] += 1

    # Anomalous sessions: unusual patterns
    for i in range(n_sessions - n_anomalous, n_sessions):
        n_events = np.random.randint(3, 50)
        # Anomalous sessions use rare templates
        rare_templates = np.random.choice(
            range(n_templates // 3, n_templates), n_events, replace=True
        )
        for t in rare_templates:
            X[i][t] += 1
        # Also add some error-like spikes
        X[i][np.random.randint(n_templates // 2, n_templates)] += np.random.randint(5, 20)

    y = np.concatenate([
        np.zeros(n_sessions - n_anomalous),
        np.ones(n_anomalous)
    ]).astype(int)

    # Shuffle
    idx = np.random.permutation(n_sessions)
    X, y = X[idx], y[idx]

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def generate_synthetic_bgl_data(n_samples=50000, n_templates=80):
    """Generate synthetic BGL-like data."""
    print(f"[INFO] Generating synthetic BGL data: {n_samples} samples")
    np.random.seed(123)

    anomaly_rate = 0.074  # 7.4% matches real BGL
    n_anomalous = int(n_samples * anomaly_rate)

    X = np.zeros((n_samples, n_templates))

    for i in range(n_samples - n_anomalous):
        n_events = np.random.randint(10, 40)
        templates = np.random.choice(n_templates // 2, n_events, replace=True)
        for t in templates:
            X[i][t] += 1

    for i in range(n_samples - n_anomalous, n_samples):
        n_events = np.random.randint(5, 60)
        templates = np.random.choice(range(n_templates // 3, n_templates), n_events, replace=True)
        for t in templates:
            X[i][t] += 1
        X[i][np.random.randint(n_templates // 2, n_templates)] += np.random.randint(10, 30)

    y = np.concatenate([
        np.zeros(n_samples - n_anomalous),
        np.ones(n_anomalous)
    ]).astype(int)

    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# ============================================================
# CLASSIFIERS
# ============================================================

def run_classifier(clf_name, X_train, y_train, X_test, y_test, seed=42):
    """Run a single classifier and return metrics."""
    classifiers = {
        'LR': LogisticRegression(max_iter=1000, random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100, random_state=seed),
        'SVM': SVC(kernel='rbf', probability=True, random_state=seed, max_iter=5000),
        'IF': IsolationForest(contamination=0.05, random_state=seed)
    }

    clf = classifiers[clf_name]

    start = time.time()
    if clf_name == 'IF':
        clf.fit(X_train)
        y_pred = (clf.predict(X_test) == -1).astype(int)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    train_time = time.time() - start

    # Inference timing (batch of 100)
    start = time.time()
    for _ in range(10):
        if clf_name == 'IF':
            _ = clf.predict(X_test[:100])
        else:
            _ = clf.predict(X_test[:100])
    inference_ms = (time.time() - start) / 10 / 100 * 1000

    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'train_time_s': float(train_time),
        'inference_ms': float(inference_ms)
    }

    try:
        if clf_name == 'IF':
            y_scores = -clf.score_samples(X_test)
        else:
            y_scores = clf.predict_proba(X_test)[:, 1]
        metrics['auc'] = float(roc_auc_score(y_test, y_scores))
    except Exception:
        metrics['auc'] = 0.0

    return metrics


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 60)
    print("Paper 1: Log Anomaly Detection - Traditional ML Experiments")
    print("=" * 60)

    datasets = {
        'HDFS': generate_synthetic_hdfs_data(n_sessions=5000),
        'BGL': generate_synthetic_bgl_data(n_samples=50000),
    }

    all_results = {}

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {ds_name}")
        print(f"Samples: {len(X)}, Features: {X.shape[1]}, "
              f"Anomaly rate: {y.mean()*100:.1f}%")
        print(f"{'=' * 50}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        ds_results = {}

        for clf_name in ['LR', 'RF', 'SVM', 'IF']:
            print(f"\n--- Drain + {clf_name} ---")

            # Run with multiple seeds
            seed_results = []
            for seed in RANDOM_SEEDS:
                metrics = run_classifier(clf_name, X_train, y_train, X_test, y_test, seed)
                seed_results.append(metrics)

            # Compute mean and std
            avg = {
                k: float(np.mean([r[k] for r in seed_results]))
                for k in seed_results[0]
            }
            std = {
                f"{k}_std": float(np.std([r[k] for r in seed_results]))
                for k in seed_results[0]
            }

            combined = {**avg, **std}
            ds_results[f"Drain+{clf_name}"] = combined

            print(f"  F1: {avg['f1']:.4f} (+/- {std['f1_std']:.4f})")
            print(f"  Precision: {avg['precision']:.4f}, Recall: {avg['recall']:.4f}")
            print(f"  AUC: {avg['auc']:.4f}")
            print(f"  Inference: {avg['inference_ms']:.3f} ms/sample")

        all_results[ds_name] = ds_results

    # Label efficiency experiment
    print(f"\n{'=' * 50}")
    print("Label Efficiency Experiment (HDFS)")
    print(f"{'=' * 50}")

    X_hdfs, y_hdfs = datasets['HDFS']
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_hdfs, y_hdfs, test_size=0.2, random_state=42, stratify=y_hdfs
    )

    label_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    label_efficiency = {}

    for frac in label_fractions:
        n_samples = max(10, int(len(X_train_full) * frac))
        X_sub = X_train_full[:n_samples]
        y_sub = y_train_full[:n_samples]

        metrics = run_classifier('RF', X_sub, y_sub, X_test, y_test, seed=42)
        label_efficiency[f"{frac*100:.0f}%"] = metrics
        print(f"  {frac*100:5.1f}% labels ({n_samples:5d} samples): F1 = {metrics['f1']:.4f}")

    all_results['label_efficiency'] = label_efficiency

    # Save results
    output_file = RESULTS_DIR / "traditional_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] Results -> {output_file}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'HDFS F1':>10} {'BGL F1':>10} {'Latency(ms)':>12}")
    print("-" * 52)
    for method in ['Drain+LR', 'Drain+RF', 'Drain+SVM', 'Drain+IF']:
        hdfs_f1 = all_results['HDFS'][method]['f1'] * 100
        bgl_f1 = all_results['BGL'][method]['f1'] * 100
        latency = all_results['HDFS'][method]['inference_ms']
        print(f"{method:<20} {hdfs_f1:>9.1f}% {bgl_f1:>9.1f}% {latency:>11.3f}")

    print(f"\n{'=' * 70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
