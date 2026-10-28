"""Validation for the opt-in study, independent of benchmark configuration."""

from copy import deepcopy
import math
from pathlib import Path
import re
import yaml
from .config import validate_config


def validate_optimized(config: dict) -> None:
    # Reuse benchmark scalar validation with a compatibility view, without changing it.
    view = deepcopy(config)
    view["neural"]["bidirectional"] = False
    view["neural"]["dropout"] = config["neural"]["classifier_dropout"]
    view["baselines"] = {"max_features": 1, "min_df": 1, "logistic_max_iter": 1,
                         "c_grid": [1.0], "ngram_range": [1, 1]}
    validate_config(view)
    data, net = config["data"], config["neural"]
    if data["truncation"] not in ("head", "head_tail"):
        raise ValueError("truncation must be head or head_tail")
    if net["pooling"] not in ("final", "max", "mean_max"):
        raise ValueError("pooling must be final, max or mean_max")
    if not isinstance(net["bidirectional"], bool):
        raise ValueError("bidirectional must be boolean")
    for key in ("embedding_dropout", "classifier_dropout", "word_dropout"):
        if isinstance(net[key], bool) or not math.isfinite(net[key]) or not 0 <= net[key] < 1:
            raise ValueError(f"{key} must be finite and in [0,1)")
    if net["scheduler"] not in ("none", "exponential"):
        raise ValueError("scheduler must be none or exponential")
    if not math.isfinite(net["scheduler_gamma"]) or not 0 < net["scheduler_gamma"] <= 1:
        raise ValueError("scheduler_gamma must be in (0,1]")
    if config["seed"] != 42:
        raise ValueError("This registered study requires seed 42")


def load_study_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        validate_optimized(config)
        experiments = config["experiments"]
        ids = [item["id"] for item in experiments]
        if len(ids) != 16 or len(set(ids)) != 16 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
            raise ValueError("Exactly 16 unique, safe experiment IDs are required")
        if [i for i, item in enumerate(experiments) if item.get("control")] != [0]:
            raise ValueError("Only the first experiment must be the original control")
        if experiments[1]["overrides"]:
            raise ValueError("The second experiment must be the unmodified anchor")
        for item in experiments:
            if set(item["overrides"]) - {"data", "neural"}:
                raise ValueError("Only data and neural experiment overrides are supported")
            candidate_config(config, item)
        if not isinstance(config["reference_run"], str):
            raise ValueError("reference_run must be a benchmark run path")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid study configuration: {exc}") from exc
    return config


def candidate_config(study: dict, item: dict, mode: str = "full", control_config: dict | None = None) -> dict:
    config = deepcopy(control_config if item.get("control") and control_config is not None else study)
    config.pop("experiments", None)
    config.pop("reference_run", None)
    if not item.get("control"):
        for section, changes in item["overrides"].items():
            if set(changes) - set(config[section]):
                raise ValueError(f"Unknown {section} override: {set(changes) - set(config[section])}")
            config[section].update(changes)
        validate_optimized(config)
    if mode == "smoke":
        config["data"].update(max_vocab_size=min(config["data"]["max_vocab_size"], 2000), max_seq_len=32)
        config["neural"].update(embedding_dim=16, hidden_dim=16, epochs=1)
    elif mode != "full":
        raise ValueError("mode must be full or smoke")
    return config
