"""Reserved CLS slots bypass the column-wise set transformer.

``ColEmbedding`` returns embeddings with ``reserve_cls_tokens`` leading slots on the
feature axis so that ``RowInteraction`` can write its CLS tokens in place. These slots
must never be computed by the column-wise set transformer: they are zero-filled and
overwritten afterwards.
"""

from collections import OrderedDict

import pytest
import torch

from tabicl._model import inference
from tabicl._model.inference import InferenceManager, OffloadMode
from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL

SMALL = dict(
    embed_dim=16,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=4,
    row_num_blocks=1,
    row_nhead=2,
    icl_num_blocks=1,
    icl_nhead=2,
    zero_init=False,
)


class _FakeAccelerator(str):
    """A CPU device that ``InferenceManager`` does not treat as CPU, so it takes the batched path."""

    type = "fake"
    index = None


def _make_model(**kwargs):
    torch.manual_seed(0)
    return TabICL(**{**SMALL, **kwargs})


@pytest.mark.parametrize("feature_group", ["same", "valid", False])
@pytest.mark.parametrize("training", [True, False])
def test_column_transformer_never_sees_reserved_slots(feature_group, training):
    model = _make_model(col_feature_group=feature_group).train(training)
    col_embedder = model.col_embedder
    B, T, H, C = 2, 12, 5, col_embedder.reserve_cls_tokens
    X = torch.randn(B, T, H)
    y_train = torch.arange(8).remainder(2).expand(B, -1)

    num_cols = []
    hook = col_embedder.tf_col.register_forward_pre_hook(lambda mod, args, kwargs: num_cols.append(args[0].shape[-3]), with_kwargs=True)
    try:
        with torch.set_grad_enabled(training):
            embeddings = col_embedder(X, y_train)
    finally:
        hook.remove()

    G = embeddings.shape[2] - C
    assert G == (2 if feature_group == "valid" else H)
    assert num_cols and all(n == G for n in num_cols)
    assert torch.all(embeddings[:, :, :C] == 0)


@pytest.mark.parametrize("feature_group", ["same", False])
@pytest.mark.parametrize("offload", [OffloadMode.GPU, OffloadMode.CPU])
def test_batched_inference_places_outputs_after_reserved_slots(monkeypatch, feature_group, offload):
    model = _make_model(col_feature_group=feature_group).eval()
    col_embedder = model.col_embedder
    B, T, H, C, train_size = 3, 12, 5, col_embedder.reserve_cls_tokens, 8
    X = torch.randn(B, T, H)
    y_train = torch.arange(train_size).remainder(2).expand(B, -1)
    config = dict(device="cpu", use_amp=False, use_fa3=False, use_async=False, min_batch_size=1)

    with torch.no_grad():
        expected = col_embedder(X, y_train, mgr_config=config)
        cache = KVCache()
        col_embedder.forward_with_cache(X[:, :train_size], cache, y_train=y_train, mgr_config=config)
        expected_cached = col_embedder.forward_with_cache(
            X[:, train_size:], cache, use_cache=True, store_cache=False, mgr_config=config
        )

    # Force the multi-batch path with a batch size of 2 over B * G column sets
    monkeypatch.setattr(inference, "resolve_torch_device", lambda device: _FakeAccelerator("cpu"))
    monkeypatch.setattr(InferenceManager, "estimate_safe_batch_size", lambda self, *args, **kwargs: (1.0, 2))
    monkeypatch.setattr(InferenceManager, "_resolve_offload_mode", lambda self, *args: (offload, None))
    config["max_pinned_memory_mb"] = 0

    with torch.no_grad():
        actual = col_embedder(X, y_train, mgr_config=config)
        cache = KVCache()
        col_embedder.forward_with_cache(X[:, :train_size], cache, y_train=y_train, mgr_config=config)
        actual_cached = col_embedder.forward_with_cache(
            X[:, train_size:], cache, use_cache=True, store_cache=False, mgr_config=config
        )

    assert torch.all(actual[:, :, :C] == 0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_cached, expected_cached)


def test_inference_manager_reserve_on_cpu_path():
    mgr = InferenceManager(enc_name="tf_col", out_dim=4)
    mgr.configure(device="cpu", use_amp=False, use_fa3=False)
    features = torch.randn(2, 3, 8, 1)

    out = mgr(lambda features: features.expand(-1, -1, -1, 4) + 1, OrderedDict([("features", features)]), reserve=2)

    assert out.shape == (2, 5, 8, 4)
    assert torch.all(out[:, :2] == 0)
    torch.testing.assert_close(out[:, 2:], features.expand(-1, -1, -1, 4) + 1)


@pytest.mark.parametrize("feature_group", ["same", False])
def test_column_cache_with_reserved_slot_entries_is_still_usable(feature_group):
    """Caches stored before this change hold K/V for the reserved slots too (e.g. saved estimators)."""
    model = _make_model(col_feature_group=feature_group).eval()
    col_embedder = model.col_embedder
    B, T, H, C, train_size = 2, 12, 5, col_embedder.reserve_cls_tokens, 8
    X = torch.randn(B, T, H)
    y_train = torch.arange(train_size).remainder(2).expand(B, -1)
    config = dict(device="cpu", use_amp=False, use_fa3=False)

    with torch.no_grad():
        cache = KVCache()
        col_embedder.forward_with_cache(X[:, :train_size], cache, y_train=y_train, mgr_config=config)
        expected = col_embedder.forward_with_cache(
            X[:, train_size:], cache, use_cache=True, store_cache=False, mgr_config=config
        )

        # Old layout: C extra entries ahead of the real columns
        for entry in cache.kv.values():
            entry.key = torch.cat([torch.randn_like(entry.key[:, :C]), entry.key], dim=1)
            entry.value = torch.cat([torch.randn_like(entry.value[:, :C]), entry.value], dim=1)
        actual = col_embedder.forward_with_cache(
            X[:, train_size:], cache, use_cache=True, store_cache=False, mgr_config=config
        )

    torch.testing.assert_close(actual, expected)
