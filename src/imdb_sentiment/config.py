"""Load and validate experiment settings before any data access."""

from copy import deepcopy
from pathlib import Path
import math
import yaml


def validate_config(config: dict) -> None:
    """Reject invalid values and unsupported architectures early."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    try:
        data, base, net = (config[k] for k in ("data", "baselines", "neural"))
        for section, keys in [(data, ["max_vocab_size", "max_seq_len", "batch_size"]),
                              (base, ["max_features", "min_df", "logistic_max_iter"]),
                              (net, ["embedding_dim", "hidden_dim", "num_layers", "epochs", "early_stopping_patience"])]:
            for key in keys:
                value = section[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
        if data["max_vocab_size"] < 3:
            raise ValueError("max_vocab_size must include PAD, UNK and at least one word")
        if data["validation_fraction"] != 0.2:
            raise ValueError("This experiment requires validation_fraction=0.2")
        if data["dataset_name"] != "stanfordnlp/imdb":
            raise ValueError("This experiment requires stanfordnlp/imdb")
        if not isinstance(data["lowercase"], bool):
            raise ValueError("lowercase must be boolean")
        workers = data.get("num_workers", 0)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or not 0 <= config["seed"] < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32-1")
        if net["bidirectional"] is not False:
            raise ValueError("Only unidirectional models are supported")
        if net["optimizer"] != "adam" or net["selection_metric"] != "val_loss":
            raise ValueError("Use optimizer=adam and selection_metric=val_loss")
        if not math.isfinite(net["dropout"]) or not 0 <= net["dropout"] < 1:
            raise ValueError("dropout must be in [0,1)")
        for key in ["learning_rate", "grad_clip"]:
            if not math.isfinite(net[key]) or net[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ["weight_decay", "early_stopping_min_delta"]:
            if not math.isfinite(net[key]) or net[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not base["c_grid"] or any(not math.isfinite(c) or c <= 0 for c in base["c_grid"]):
            raise ValueError("c_grid must contain finite positive values")
        ngrams = base["ngram_range"]
        if len(ngrams) != 2 or any(not isinstance(n, int) for n in ngrams) or not 1 <= ngrams[0] <= ngrams[1]:
            raise ValueError("Invalid ngram_range")
        if config.get("device", "auto") not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
            raise ValueError("output_dir must be a nonempty path")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid or missing configuration field: {exc}") from exc


def load_config(path: str | Path, mode: str = "full") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return effective_config(config, mode)


def effective_config(config: dict, mode: str) -> dict:
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    config = deepcopy(config)
    if mode == "smoke":
        config["data"].update(max_seq_len=64, max_vocab_size=2000)
        config["baselines"].update(max_features=5000, min_df=1)
        config["neural"]["epochs"] = 1
    validate_config(config)
    return config
