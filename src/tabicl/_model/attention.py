from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3

    HAS_FLASH_ATTN3 = True
except ImportError:
    HAS_FLASH_ATTN3 = False

_use_flash_attn3 = True


@contextmanager
def flash_attn3_toggle(enabled: bool):
    """Context manager to temporarily enable or disable Flash Attention 3.

    Used by ``InferenceManager._run_forward()`` in ``inference.py`` to control
    whether Flash Attention 3 is used during each forward pass based on the
    ``use_fa3`` configuration.
    """
    global _use_flash_attn3
    old = _use_flash_attn3
    _use_flash_attn3 = enabled
    try:
        yield
    finally:
        _use_flash_attn3 = old


def set_flash_attn3_enabled(enabled: bool):
    """Globally enable or disable Flash Attention 3 for this process.

    Used by the pre-training ``Trainer`` (``--use_flash_attn3``): FA3 runs attention
    in fp16, so the TabICLv2 recipe enables it only for stages 2 and 3.
    """
    global _use_flash_attn3
    _use_flash_attn3 = enabled


def sdpa_with_flattened_batch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None,
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's ``scaled_dot_product_attention`` and then reshaping the
    output back to the original shape. This flattening is necessary to properly
    trigger Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:

        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len).

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input.
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

    if ssmax_layer is not None:
        src_len = k.size(-2)
        q = ssmax_layer(q, src_len)

    # FlashAttn3 doesn't support dropout and custom attention mask
    if HAS_FLASH_ATTN3 and _use_flash_attn3 and q.is_cuda and attn_mask is None and dropout_p == 0.0:
        # FlashAttention only supports fp16, bf16, and fp8_e4m3
        # Convert to bf16 if needed, then convert back to original dtype
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            fa_dtype = torch.float16
        else:
            fa_dtype = orig_dtype

        flat_bs, nheads, seqlen_q, headdim = q.shape
        seqlen_k = k.shape[-2]
        q_fa = q.transpose(1, 2).reshape(flat_bs * seqlen_q, nheads, headdim).contiguous().to(fa_dtype)
        k_fa = k.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        v_fa = v.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        cu_seqlens_q = torch.arange(0, (flat_bs + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (flat_bs + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        out = flash_attn3(q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        out = out.view(flat_bs, seqlen_q, nheads, headdim).transpose(1, 2).to(orig_dtype)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    need_kv: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Multi-head attention with support for rotary position embeddings.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim).

    num_heads : int
        Number of attention heads.

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections.

    in_proj_bias : Tensor
        Combined bias vector for input projections.

    dropout_p : float
        Dropout probability applied to attention weights.

    out_proj_weight : Tensor
        Output projection weight matrix.

    out_proj_bias : Tensor
        Output projection bias vector.

    key : Optional[Tensor], default=None
        Key tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    value : Optional[Tensor], default=None
        Value tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    cached_kv : Optional[KVCacheEntry], default=None
        Pre-computed key and value projections for caching. When provided:

        - key and value parameters are ignored
        - Only query projection is computed
        - cached_kv.key shape: (..., num_heads, src_len, head_dim)
        - cached_kv.value shape: (..., num_heads, src_len, head_dim)
        - RoPE is applied only to queries (keys should already have RoPE applied)

    training : bool, default=True
        Whether the model is in training mode (affects dropout).

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:

        - For binary masks: True values indicate positions to ignore.
        - For float masks: Values are directly added to attention scores.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (tgt_len, src_len) or
        (..., num_heads, tgt_len, src_len).

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    need_kv : bool, default=False
        If True and ``cached_kv`` is None, also returns the computed K and V
        projections along with the attention output. Useful for caching K/V for
        subsequent calls.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        If ``need_kv`` is False or ``cached_kv`` is provided:
            Attention output tensor of shape (..., tgt_len, embed_dim).
        If ``need_kv`` is True and ``cached_kv`` is None:
            Tuple of (attn_output, k, v) where:

            - attn_output: shape (..., tgt_len, embed_dim)
            - k: shape (..., num_heads, src_len, head_dim)
            - v: shape (..., num_heads, src_len, head_dim)
    """

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    if cached_kv is None:
        # Standard: project Q, K, V jointly
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        src_len = key.shape[-2]
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
    else:
        # Use cached K/V, project Q only
        k, v = cached_kv.key, cached_kv.value
        src_len = k.shape[-2]
        q_proj_weight = in_proj_weight[:embed_dim]
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    # Process attention mask
    correct_2d_shape = (tgt_len, src_len)
    correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != correct_2d_shape:
                raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
            attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
        elif attn_mask.dim() == len(correct_nd_shape):
            if attn_mask.shape != correct_nd_shape:
                raise ValueError(
                    f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                    f"but got {attn_mask.shape}"
                )
        else:
            raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

    # Process key padding mask
    if key_padding_mask is not None:
        if key_padding_mask.shape != (*batch_shape, src_len):
            raise ValueError(
                f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
            )
        key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
            *batch_shape, num_heads, tgt_len, src_len
        )

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    attn_output = sdpa_with_flattened_batch(
        q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer
    )  # (..., nh, tgt_len, hs)

    # Reshape and project output
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    if need_kv and cached_kv is None:
        return attn_output, k, v

    return attn_output


def chunked_kv_attention_forward(
    query: Tensor,
    key_value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    chunk_size: int,
    kv_norm: Optional[nn.Module] = None,
    ssmax_layer: Optional[nn.Module] = None,
) -> Tensor:
    """Multi-head attention over a long key/value sequence, a chunk at a time.

    Equivalent to ``multi_head_attention_forward(query, key=kv, value=kv)`` in
    eval mode without masks or RoPE, where ``kv = kv_norm(key_value)``, but
    keys and values are projected and attended to ``chunk_size`` positions at
    a time with an online softmax. Memory thus scales with ``chunk_size``
    instead of ``src_len``, apart from ``key_value`` itself. The softmax is
    accumulated in float32.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim).

    key_value : Tensor
        Key and value tensor of shape (..., src_len, embed_dim).

    num_heads : int
        Number of attention heads.

    in_proj_weight, in_proj_bias : Tensor
        Packed Q, K, V input projection.

    out_proj_weight, out_proj_bias : Tensor
        Output projection.

    chunk_size : int
        Number of key/value positions processed at once.

    kv_norm : Optional[nn.Module], default=None
        Normalization applied to each chunk of ``key_value`` before projection.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries, with
        ``src_len`` as the sequence length.

    Returns
    -------
    Tensor
        Attention output of shape (..., tgt_len, embed_dim).
    """
    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key_value.shape[-2]
    head_dim = embed_dim // num_heads
    w_q, w_k, w_v = in_proj_weight.chunk(3)
    b_q, b_k, b_v = in_proj_bias.chunk(3) if in_proj_bias is not None else (None, None, None)

    def split_heads(x: Tensor) -> Tensor:
        return x.view(*x.shape[:-1], num_heads, head_dim).transpose(-3, -2)

    q = split_heads(F.linear(query, w_q, b_q))  # (..., nh, tgt_len, hs)
    if ssmax_layer is not None:
        q = ssmax_layer(q.reshape(-1, *q.shape[-3:]), src_len).view(q.shape)
    out_dtype = q.dtype
    q = q.float() * head_dim**-0.5

    running_max = q.new_full((*q.shape[:-1], 1), float("-inf"))
    running_sum = q.new_zeros((*q.shape[:-1], 1))
    acc = torch.zeros_like(q)
    for start in range(0, src_len, chunk_size):
        # Normalization and projections run under the caller's autocast settings,
        # only the softmax accumulation is forced to float32
        kv = key_value[..., start : start + chunk_size, :]
        if kv_norm is not None:
            kv = kv_norm(kv)
        k = split_heads(F.linear(kv, w_k, b_k))
        v = split_heads(F.linear(kv, w_v, b_v))
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = q @ k.float().transpose(-2, -1)  # (..., nh, tgt_len, chunk)
            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            weights = torch.exp(scores - new_max)
            rescale = torch.exp(running_max - new_max)
            running_sum = running_sum * rescale + weights.sum(dim=-1, keepdim=True)
            acc = acc * rescale + weights @ v.float()
            running_max = new_max
        del kv, k, v, scores, weights

    attn_output = (acc / running_sum).to(out_dtype)
    attn_output = attn_output.transpose(-3, -2).reshape(*batch_shape, tgt_len, embed_dim)
    return F.linear(attn_output, out_proj_weight, out_proj_bias)
