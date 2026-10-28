import pytest
import torch
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from imdb_sentiment.config import load_config
from imdb_sentiment.data import load_data


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path):
    config = load_config("configs/default.yaml", "smoke")
    config["output_dir"] = str(tmp_path / "outputs")
    config["device"] = "cpu"
    config["neural"].update(embedding_dim=8, hidden_dim=8)
    return config


@pytest.fixture
def toy_dataset():
    features = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    def make(size, prefix):
        return Dataset.from_dict({"text": [f"{prefix} {i} " + ("awful boring film never good" if i % 2 == 0 else "wonderful enjoyable film not boring") for i in range(size)],
                                  "label": [i % 2 for i in range(size)]}, features=features)
    return DatasetDict(train=make(40, "training"), test=make(20, "heldoutword"))


@pytest.fixture
def bundle(config, toy_dataset):
    return load_data(config, "smoke", dataset=toy_dataset)
