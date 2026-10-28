"""Validation-only selection of sparse linear estimators."""

import time
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from .evaluation import compute_metrics, save_json


def candidate_key(row: dict) -> tuple:
    return row["f1"], row["accuracy"], -row["C"]


def train_baseline(name: str, features, bundle, config: dict, directory):
    start = time.perf_counter()
    candidates, best_key, best_model, best_metrics = [], None, None, None
    for c in sorted(set(config["baselines"]["c_grid"])):
        estimator = (LogisticRegression(C=c, solver="liblinear", max_iter=config["baselines"]["logistic_max_iter"], random_state=config["seed"])
                     if name == "logreg" else LinearSVC(C=c, dual="auto", random_state=config["seed"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(features.train, bundle.train.label)
        messages = [str(w.message) for w in caught]
        for message in messages:
            print(f"WARNING [{name}, C={c}]: {message}", flush=True)
        metrics = compute_metrics(bundle.val.label, estimator.predict(features.val))
        row = {"C": c, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}, "warnings": " | ".join(messages)}
        candidates.append(row)
        if best_key is None or candidate_key(row) > best_key:
            best_key, best_model, best_metrics = candidate_key(row), estimator, metrics
        print(f"{name}: C={c}, validation F1={metrics['f1']:.4f}", flush=True)
    seconds = time.perf_counter() - start
    pd.DataFrame(candidates).to_csv(directory / "hyperparameter_search.csv", index=False)
    save_json(directory / "validation_metrics.json", best_metrics)
    joblib.dump(best_model, directory / "estimator.joblib")
    return best_model, {"best_hyperparameters": {"C": float(best_model.C)}, "training_seconds": seconds,
                        "parameter_count": int(best_model.coef_.size + best_model.intercept_.size),
                        "timing_scope": "All C candidates: fitting and validation; excludes shared TF-IDF preparation and test evaluation"}
