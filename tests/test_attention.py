"""Attention splitting for devices with a per-call limit on attention scores.

On XPU, SDPA is split over batch and queries so that no call exceeds
``_MAX_SDPA_SCORES``; the tests lower that limit to check the split result.
"""

import pytest
import torch
from torch.nn import functional as F

from tabicl._model import attention
from tests.torch_devices_helpers import skip_if_device_unusable, DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_sdpa_split_over_batch_and_queries(device, monkeypatch):
    skip_if_device_unusable(device)
    # 2 heads x 3 keys = 6 scores per query: a limit of 12 gives 1 table x 2 queries per call.
    monkeypatch.setitem(attention._MAX_SDPA_SCORES, torch.device(device).type, 12)
    torch.manual_seed(0)
    q = torch.randn(2, 2, 5, 4, device=device)
    k = torch.randn(2, 2, 3, 4, device=device)
    v = torch.randn(2, 2, 3, 4, device=device)
    mask = torch.randn(2, 2, 5, 3, device=device)

    out = attention.sdpa_with_flattened_batch(q, k, v, mask)

    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, mask))
