from copy import deepcopy
import pytest
import torch
from imdb_sentiment.optimization_config import load_study_config
from imdb_sentiment.optimized_lstm import OptimizedLSTM, masked_pool, final_hidden
from imdb_sentiment.optimization_data import truncate_tokens, OptimizedReviewDataset


@pytest.fixture
def optimized_config():
    config = load_study_config("configs/lstm_ablation.yaml")
    config["neural"].update(embedding_dim=4, hidden_dim=5, embedding_dropout=0.0, classifier_dropout=0.0)
    return config


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pooling", ["final", "max", "mean_max"])
@pytest.mark.parametrize("layers", [1, 2])
def test_optimized_shapes_padding_and_gradients(optimized_config, bidirectional, pooling, layers):
    config = optimized_config
    config["neural"].update(bidirectional=bidirectional, pooling=pooling, num_layers=layers)
    model = OptimizedLSTM(10, config).eval()
    ids = torch.tensor([[2, 3, 0], [4, 0, 0], [5, 6, 7]])
    lengths = torch.tensor([2, 1, 3])
    logits = model(ids, lengths)
    assert logits.shape == (3,) and torch.isfinite(logits).all()
    padded = torch.cat([ids, torch.zeros(3, 4, dtype=torch.long)], dim=1)
    assert torch.allclose(logits, model(padded, lengths), atol=1e-6)
    for i, length in enumerate(lengths):
        individual = model(ids[i:i+1, :length], lengths[i:i+1])
        assert individual.shape == (1,) and torch.allclose(individual[0], logits[i], atol=1e-6)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 1.])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_directional_final_state_order():
    hidden = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    expected = torch.cat([hidden[2], hidden[3]], dim=1)
    assert torch.equal(final_hidden(hidden, 2, True), expected)
    assert torch.equal(final_hidden(hidden[:2], 2, False), hidden[1])


def test_mask_negative_outputs():
    outputs = torch.tensor([[[-5., -7.], [-2., -4.], [1000., 1000.]], [[-9., -3.], [0., 0.], [0., 0.]]])
    lengths = torch.tensor([2, 1])
    maximum = masked_pool(outputs, lengths, "max")
    assert torch.equal(maximum, torch.tensor([[-2., -4.], [-9., -3.]]))
    both = masked_pool(outputs, lengths, "mean_max")
    assert torch.equal(both[:, :2], torch.tensor([[-3.5, -5.5], [-9., -3.]]))
    assert torch.equal(both[:, 2:], maximum)


@pytest.mark.parametrize("limit,expected", [(1, [0]), (3, [0, 1, 9]), (4, [0, 1, 8, 9]), (10, list(range(10))), (20, list(range(10)))])
def test_head_tail_limits(limit, expected):
    assert truncate_tokens(list(range(10)), limit, "head_tail") == expected
    assert truncate_tokens(list(range(10)), limit, "head") == list(range(10))[:limit]


def test_dropout_preserves_inputs_and_eval(optimized_config):
    config = optimized_config
    config["neural"].update(word_dropout=1.0, embedding_dropout=.5, classifier_dropout=.5)
    model = OptimizedLSTM(10, config)
    ids = torch.tensor([[0, 1, 2, 3]])
    original = ids.clone()
    assert torch.equal(model.drop_words(ids), torch.tensor([[0, 1, 1, 1]]))
    assert torch.equal(ids, original)
    model.eval()
    assert torch.equal(model.drop_words(ids), ids)
    valid = torch.tensor([[2, 3, 0]])
    lengths = torch.tensor([2])
    assert torch.equal(model(valid, lengths), model(valid, lengths))
    model.train()
    assert not torch.equal(model.embedding_dropout(torch.ones(1000)), torch.ones(1000))


def test_empty_input(optimized_config):
    import pandas as pd
    dataset = OptimizedReviewDataset(pd.DataFrame({"text": [""], "label": [1]}), {"<PAD>": 0, "<UNK>": 1}, optimized_config)
    assert dataset[0][0].tolist() == [1]
