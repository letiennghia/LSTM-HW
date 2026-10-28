import pytest
from datasets import DatasetDict, concatenate_datasets
from imdb_sentiment.data import clean_text, load_data
from imdb_sentiment.config import validate_config, effective_config


def test_cleaning_retains_negation():
    assert clean_text("  NOT<br />good &amp; I don't like it.\n Never! ") == "not good & i don't like it. never!"
    assert clean_text("isn't <b>bad</b> &lt;br /&gt; no") == "isn't bad no"


def test_split_repeatability_and_identity(config, toy_dataset):
    one = load_data(config, "smoke", toy_dataset)
    two = load_data(config, "smoke", toy_dataset)
    assert one.manifest().equals(two.manifest())
    assert not set(one.train.review_id) & set(one.val.review_id)
    assert one.manifest().review_id.is_unique
    assert len(one.train) == 32 and len(one.val) == 8 and len(one.test) == 20
    assert one.train.label.value_counts().to_dict() == {0: 16, 1: 16}


def test_realistic_smoke_sizes(config, toy_dataset):
    source = DatasetDict(train=concatenate_datasets([toy_dataset["train"]] * 25),
                         test=concatenate_datasets([toy_dataset["test"]] * 10))
    bundle = load_data(config, "smoke", source)
    assert [len(frame) for _, frame in bundle.splits()] == [256, 64, 128]
    for _, frame in bundle.splits():
        assert frame.label.value_counts()[0] == frame.label.value_counts()[1]


@pytest.mark.parametrize("section,key,value", [("data", "batch_size", 0), ("data", "max_seq_len", -1),
    ("data", "max_vocab_size", 2), ("data", "num_workers", -1), ("data", "validation_fraction", 0.3),
    ("neural", "num_layers", 0), ("neural", "bidirectional", True), ("neural", "dropout", 1),
    ("neural", "learning_rate", float("nan")), ("baselines", "c_grid", [0])])
def test_invalid_configuration(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_smoke_does_not_mutate_full_config():
    from imdb_sentiment.config import load_config
    full = load_config("configs/default.yaml")
    smoke = effective_config(full, "smoke")
    assert full["data"]["max_seq_len"] == 300
    assert smoke["data"]["max_seq_len"] == 64
    assert smoke["neural"]["epochs"] == 1
