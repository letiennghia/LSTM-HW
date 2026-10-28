import pytest
import torch
from imdb_sentiment.models import SentimentRNN, SentimentLSTM
from imdb_sentiment.trainer import EarlyStopping, evaluate, train_neural
from imdb_sentiment.features import build_vocab, make_loaders


@pytest.mark.parametrize("cls", [SentimentRNN, SentimentLSTM])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_forward_and_optimizer(cls, batch_size):
    model = cls(10, embedding_dim=4, hidden_dim=5, dropout=0)
    ids = torch.tensor([[2, 3, 0], [4, 0, 0], [5, 6, 7]])[:batch_size]
    lengths = torch.tensor([2, 1, 3])[:batch_size]
    result = model(ids, lengths)
    assert result.shape == (batch_size,) and torch.isfinite(result).all()
    before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    torch.nn.BCEWithLogitsLoss()(result, torch.ones(batch_size)).backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    assert torch.equal(model.embedding.weight[0], torch.zeros(4))
    model.eval()
    padded = torch.cat([ids, torch.zeros(batch_size, 2, dtype=torch.long)], dim=1)
    assert torch.allclose(model(ids, lengths), model(padded, lengths))


def test_early_stopping():
    stop = EarlyStopping(2, 0.0001)
    assert stop.update(0.5) == (True, False)
    assert stop.update(0.49999) == (False, False)
    assert stop.update(0.51) == (False, True)
    with pytest.raises(RuntimeError):
        stop.update(float("nan"))


def test_evaluation_is_sample_weighted():
    class Constant(torch.nn.Module):
        def forward(self, ids, lengths):
            return ids[:, 0].float()
    loader = [(torch.tensor([[0], [0]]), torch.tensor([1, 1]), torch.tensor([0., 1.])),
              (torch.tensor([[3]]), torch.tensor([1]), torch.tensor([0.]))]
    loss, _, _ = evaluate(Constant(), loader, torch.device("cpu"))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0., 0., 3.]), torch.tensor([0., 1., 0.])).item()
    assert loss == pytest.approx(expected)


def test_restores_saved_checkpoint(config, bundle, tmp_path):
    config["neural"]["epochs"] = 3
    config["neural"]["early_stopping_min_delta"] = 10.0
    vocab = build_vocab(bundle.train.text, 2000)
    model = SentimentRNN(len(vocab), embedding_dim=8, hidden_dim=8)
    model, metadata = train_neural(model, make_loaders(bundle, vocab, config), config, tmp_path, torch.device("cpu"))
    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=True)
    assert metadata["best_epoch"] == 1 and metadata["checkpoint_restored"]
    assert all(torch.equal(value, checkpoint["state_dict"][key]) for key, value in model.state_dict().items())
