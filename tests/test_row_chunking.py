"""Row-chunked inference must reproduce the unchunked forward passes.

The models are small and randomly initialized (with ``zero_init=False`` so that
no block is an identity), which is enough to check that embedding the table a
chunk of rows at a time computes the same thing as embedding it at once.
"""

from __future__ import annotations

import pytest
import torch

from tabicl import TabICLClassifier
from tabicl._model.embedding import ColEmbedding
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL
from tests.torch_devices_helpers import DEVICES, skip_if_device_unusable

TRAIN_SIZE = 20
TEST_SIZE = 13
N_FEATURES = 5
# Chunk sizes straddling the train/test boundary, aligned with it, and covering one row
CHUNK_SIZES = [1, 7, 10, TRAIN_SIZE]


def _make_model(max_classes: int = 4, feature_group="same") -> TabICL:
    torch.manual_seed(0)
    model = TabICL(
        max_classes=max_classes,
        num_quantiles=9,
        embed_dim=16,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=8,
        col_feature_group=feature_group,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
        zero_init=False,
    )
    return model.eval()


def _make_data(B: int = 2, num_classes: int = 3, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(B, TRAIN_SIZE + TEST_SIZE, N_FEATURES, generator=gen)
    if num_classes > 0:
        y_train = torch.randint(0, num_classes, (B, TRAIN_SIZE), generator=gen).float()
        # Every table must contain every class
        y_train[:, :num_classes] = torch.arange(num_classes).float()
    else:
        y_train = torch.randn(B, TRAIN_SIZE, generator=gen)
    return X, y_train


def _config(device: str, row_chunk_size=None) -> InferenceConfig:
    mgr = {"device": device, "use_amp": False, "use_fa3": False, "offload": "gpu"}
    return InferenceConfig(COL_CONFIG=dict(mgr), ROW_CONFIG=dict(mgr), ICL_CONFIG=dict(mgr), ROW_CHUNK_SIZE=row_chunk_size)


def _force_multi_batch(model: TabICL, monkeypatch) -> None:
    """Make the inference managers split every input on accelerators."""
    for mgr in (model.col_embedder.inference_mgr, model.row_interactor.inference_mgr):
        monkeypatch.setattr(
            mgr,
            "estimate_safe_batch_size",
            lambda seq_len, include_inputs=True, in_dim=None, max_bs=50000: (1.0, 1),
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target_aware", [True, False])
def test_col_embedding_cached_chunks_with_labels(device, target_aware, monkeypatch):
    """Labeled row chunks embedded from a cache match a single store-mode pass."""
    skip_if_device_unusable(device)
    torch.manual_seed(0)
    embedder = ColEmbedding(
        embed_dim=16,
        num_blocks=2,
        nhead=2,
        dim_feedforward=32,
        num_inds=8,
        target_aware=target_aware,
        max_classes=4,
        reserve_cls_tokens=2,
        zero_init=False,
    ).eval()
    monkeypatch.setattr(
        embedder.inference_mgr,
        "estimate_safe_batch_size",
        lambda seq_len, include_inputs=True, in_dim=None, max_bs=50000: (1.0, 1),
    )
    mgr_config = _config(device).COL_CONFIG
    X, y_train = _make_data()

    with torch.no_grad():
        expected = embedder.forward_with_cache(
            X, col_cache=KVCache(), y_train=y_train, store_cache=True, use_cache=False, mgr_config=mgr_config
        )

        cache = KVCache()
        out = embedder.forward_with_cache(
            X[:, :TRAIN_SIZE],
            col_cache=cache,
            y_train=y_train,
            store_cache=True,
            use_cache=False,
            mgr_config=mgr_config,
            discard_output=True,
        )
        assert out is None

        # Chunks: train rows only, train/test straddling, test rows only
        bounds = [(0, 12), (12, 25), (25, TRAIN_SIZE + TEST_SIZE)]
        chunks = []
        for start, end in bounds:
            y_chunk = y_train[:, start : min(end, TRAIN_SIZE)] if start < TRAIN_SIZE else None
            chunks.append(
                embedder.forward_with_cache(
                    X[:, start:end],
                    col_cache=cache,
                    y_train=y_chunk,
                    use_cache=True,
                    store_cache=False,
                    mgr_config=mgr_config,
                )
            )

    # The reserved CLS slots are skipped by the set transformer in test-only chunks
    # (all values are -100) and are overwritten by RowInteraction anyway.
    C = embedder.reserve_cls_tokens
    torch.testing.assert_close(torch.cat(chunks, dim=1)[:, :, C:].cpu(), expected[:, :, C:].cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("row_chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize(
    "max_classes, num_classes, feature_group",
    [(4, 3, "same"), (4, 3, False), (0, 0, "same")],
    ids=["classification", "classification-no-group", "regression"],
)
def test_inference_forward_chunked(device, row_chunk_size, max_classes, num_classes, feature_group, monkeypatch):
    """Chunking the rows does not change the predictions of the regular forward pass."""
    skip_if_device_unusable(device)
    model = _make_model(max_classes=max_classes, feature_group=feature_group)
    _force_multi_batch(model, monkeypatch)
    X, y_train = _make_data(num_classes=num_classes)

    with torch.no_grad():
        expected = model(X, y_train, inference_config=_config(device))
        out = model(X, y_train, inference_config=_config(device, row_chunk_size))

    torch.testing.assert_close(out.cpu(), expected.cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_inference_forward_chunked_feature_shuffles(device, monkeypatch):
    """The shuffle optimization (embed one table, permute for the others) still applies."""
    skip_if_device_unusable(device)
    model = _make_model(feature_group=False)
    _force_multi_batch(model, monkeypatch)
    X, y_train = _make_data(B=1)
    shuffles = [[0, 1, 2, 3, 4], [4, 2, 0, 1, 3], [1, 0, 3, 4, 2]]
    X = torch.cat([X[:, :, s] for s in shuffles])
    y_train = y_train.expand(len(shuffles), -1)

    with torch.no_grad():
        expected = model(X, y_train, feature_shuffles=shuffles, inference_config=_config(device))
        out = model(X, y_train, feature_shuffles=shuffles, inference_config=_config(device, 7))

    torch.testing.assert_close(out.cpu(), expected.cpu())


@pytest.mark.parametrize(
    "max_classes, num_classes, embed_with_test",
    [(3, 5, False), (4, 3, True)],
    ids=["mixed-radix", "embed-with-test"],
)
def test_inference_forward_unchunkable_falls_back(max_classes, num_classes, embed_with_test):
    """Inputs whose rows cannot be embedded independently are processed at once."""
    model = _make_model(max_classes=max_classes)
    X, y_train = _make_data(num_classes=num_classes)
    model._embed_rows_chunked = None  # would raise if called

    with torch.no_grad():
        expected = model(X, y_train, embed_with_test=embed_with_test, inference_config=_config("cpu"))
        out = model(X, y_train, embed_with_test=embed_with_test, inference_config=_config("cpu", 7))

    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cache_mode", ["kv", "repr"])
@pytest.mark.parametrize("max_classes, num_classes", [(4, 3), (0, 0)], ids=["classification", "regression"])
def test_forward_with_cache_chunked(device, cache_mode, max_classes, num_classes, monkeypatch):
    """Storing and using a cache in row chunks matches the unchunked cached passes."""
    skip_if_device_unusable(device)
    model = _make_model(max_classes=max_classes)
    _force_multi_batch(model, monkeypatch)
    X, y_train = _make_data(num_classes=num_classes)
    X_train, X_test = X[:, :TRAIN_SIZE], X[:, TRAIN_SIZE:]

    def run(row_chunk_size):
        config = _config(device, row_chunk_size)
        stored = model.forward_with_cache(
            X_train=X_train, y_train=y_train, X_test=X_test, cache_mode=cache_mode, inference_config=config
        )
        used = model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False, inference_config=config)
        return stored, used

    with torch.no_grad():
        expected_stored, expected_used = run(None)
        stored, used = run(7)

    torch.testing.assert_close(stored.cpu(), expected_stored.cpu())
    torch.testing.assert_close(used.cpu(), expected_used.cpu())


def test_chunked_column_embeddings_stay_small(monkeypatch):
    """The row interaction never receives column embeddings for more than a chunk of rows."""
    model = _make_model()
    X, y_train = _make_data()
    seen_rows = []
    row_forward = model.row_interactor.forward

    def spy(embeddings, *args, **kwargs):
        seen_rows.append(embeddings.shape[1])
        return row_forward(embeddings, *args, **kwargs)

    monkeypatch.setattr(model.row_interactor, "forward", spy)
    with torch.no_grad():
        model(X, y_train, inference_config=_config("cpu", 7))

    assert sum(seen_rows) == TRAIN_SIZE + TEST_SIZE
    assert max(seen_rows) == 7


def test_row_chunk_size_validation():
    with pytest.raises(ValueError, match="ROW_CHUNK_SIZE"):
        InferenceConfig(ROW_CHUNK_SIZE=0)

    config = InferenceConfig()
    config.update_from_dict({"ROW_CHUNK_SIZE": 128})
    assert config.ROW_CHUNK_SIZE == 128
    with pytest.raises(ValueError, match="ROW_CHUNK_SIZE"):
        config.update_from_dict({"ROW_CHUNK_SIZE": 1.5})


def test_estimator_passes_row_chunk_size_through():
    """``ROW_CHUNK_SIZE`` given in the estimator's inference_config dict is not dropped."""
    clf = TabICLClassifier(inference_config={"ROW_CHUNK_SIZE": 64, "ICL_CONFIG": {"safety_factor": 0.5}})
    clf.device_ = torch.device("cpu")
    clf._build_inference_config()
    assert clf.inference_config_.ROW_CHUNK_SIZE == 64
    assert clf.inference_config_.ICL_CONFIG.safety_factor == 0.5
