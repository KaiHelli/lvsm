"""
Multi-head attention module for self-attention and cross-attention.

Possible improvements:
- Add support for kv-caching (only needed in case of autoregressive generation).
- Add support for multi-query attention (num_kv_heads < num_q_heads).
"""

from typing import Optional
import math
import warnings
import torch
from einops import rearrange
from packaging.version import parse as parse_version

from .norm import QKNorm
from .identity import Identity


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        num_heads,
        dropout_p=0.1,
        cross_attn=False,
        bias=True,
        qk_norm=False,
        qk_exp_seq_len=None,
    ):
        """
        Multi-head attention mechanism.

        Args:
            d_model (int): Dimension of the input features.
            d_k (int): Dimension of the key and query space per head.
            d_v (int): Dimension of the value space per head.
            num_heads (int): Number of attention heads.
            dropout_p (float): Dropout probability to apply after attention weights. Default is 0.1.
            cross_attn (bool): If True, this layer will be used for cross-attention. Default is False (self-attention).
            qk_norm (bool): If True, QK normalization is applied.
            qk_exp_seq_len (int|None): If qk_norm is true, this defines the expected sequence lengths to initialize the QK normalization scaling parameter g_0.
        """
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.d_k_total = d_k * num_heads
        self.d_v_total = d_v * num_heads
        self.dropout_p = dropout_p
        self.cross_attn = cross_attn
        self.bias = bias
        self.qk_norm = qk_norm

        # Initialize softmax scale parameter
        if self.qk_norm:
            assert (
                qk_exp_seq_len is not None
            ), "To initialize QK normalization the expected sequence length is required. "
            self.qk_scale = QKNorm(qk_exp_seq_len)
            self.mha_scale = 1.0
        else:
            self.qk_scale = Identity()
            self.mha_scale = self.d_k**-0.5

        # Linear layers for projecting input to query, key, and value
        if self.cross_attn:
            self.q_proj = torch.nn.Linear(d_model, self.d_k_total, bias=self.bias)
            self.kv_proj = torch.nn.Linear(d_model, self.d_k_total + self.d_v_total, bias=self.bias)
        else:
            self.qkv_proj = torch.nn.Linear(d_model, 2 * self.d_k_total + self.d_v_total, bias=self.bias)

        self.o_proj = torch.nn.Linear(self.d_v_total, d_model, bias=self.bias)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        """
        if self.cross_attn:
            torch.nn.init.xavier_normal_(self.q_proj.weight)
            torch.nn.init.xavier_normal_(self.kv_proj.weight)
            if self.bias:
                self.q_proj.bias.data.fill_(0)
                self.kv_proj.bias.data.fill_(0)
        else:
            torch.nn.init.xavier_normal_(self.qkv_proj.weight)
            if self.bias:
                self.qkv_proj.bias.data.fill_(0)

        torch.nn.init.xavier_normal_(self.o_proj.weight)
        if self.bias:
            self.o_proj.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            x_kv (Optional[torch.Tensor]): Optional input for cross-attention of shape (batch_size, kv_seq_length, d_model).
            attn_mask (Optional[torch.Tensor]): Attention mask to apply. Should be broadcastable to (batch_size, num_heads, seq_length, kv_seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        qkv, q, kv = self.compute_q_kv(x, x_kv)
        attention_head_outputs = self.compute_attention_heads(qkv=qkv, q=q, kv=kv, causal=causal, attn_mask=attn_mask)
        output = self.o_proj(attention_head_outputs)
        return output

    def compute_attention_heads(
        self,
        qkv: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
        causal=False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute attention heads.

        Args:
            qkv (Optional[torch.Tensor]): Combined query, key, value tensor for self-attention.
            q (Optional[torch.Tensor]): Query tensor for cross-attention.
            kv (Optional[torch.Tensor]): Combined key-value tensor for cross-attention.
            attn_mask (Optional[torch.Tensor]): Optional attention mask tensor.

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_length, d_v_total).
        """
        assert (causal and attn_mask is None) or (
            not causal
        ), "Causal attention is not supported with specified attention mask."

        if parse_version(torch.__version__) >= parse_version("2.0.0"):
            # Use torch's scaled_dot_product_attention with SDP kernel in torch >= 2.0
            backends = []
            if not next(self.parameters()).is_cuda:
                if torch.cuda.is_available():
                    warnings.warn(
                        "Warning: The model is currently running on the CPU, even though CUDA is available. "
                        "Switching to SDPBackend.MATH, which may significantly impact performance. "
                        "Consider moving your model to the GPU for optimal efficiency."
                    )

                backends.append(torch.nn.attention.SDPBackend.MATH)
            else:
                # backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
                # backends.append(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
                backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

            with torch.nn.attention.sdpa_kernel(
                backends=backends,
            ):
                # Ensure the correct q, k, v setup for cross-attention or self-attention
                # Rearrange transposes (b, s, h, d) to (b, h, s, d) as that is expected in
                # torch's scaled_dot_product_attention
                if self.cross_attn and q is not None and kv is not None:
                    q = rearrange(q, "b s h d -> b h s d")
                    k, v = rearrange(kv, "b s t h d -> t b h s d").unbind(dim=0)
                elif qkv is not None:
                    q, k, v = rearrange(qkv, "b s t h d -> t b h s d").unbind(dim=0)
                else:
                    raise ValueError("Invalid inputs for attention computation.")

                # Apply QK-Norm if set, otherwise this applies the identity.
                q, k = self.qk_scale(q, k)

                attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=causal,
                    scale=self.mha_scale,
                )
                # Revert (b, h, s, d) to (b, s, h d) and flatten tokens per head
                return rearrange(attention_head_outputs, "b h s d -> b s (h d)")
        else:
            # Fallback implementation using traditional scaled dot-product attention
            if self.cross_attn and q is not None and kv is not None:
                k, v = rearrange(kv, "b s t h d -> t b s h d").unbind(dim=0)
            elif qkv is not None:
                q, k, v = rearrange(qkv, "b s t h d -> t b s h d").unbind(dim=0)
            else:
                raise ValueError("Invalid inputs for attention computation.")

            # Apply QK-Norm if set, otherwise this applies the identity.
            q, k = self.qk_scale(q, k)

            logits = torch.einsum("bnhd,bmhd->bhnm", q, k)

            logits *= self.mha_scale

            if causal:
                attn_mask = torch.tril(torch.ones(logits.shape[-2:], device=logits.device, dtype=torch.bool))

            if attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, float("-inf"))

            attn_weights = torch.nn.functional.softmax(logits, dim=-1)
            attn_weights = torch.nn.functional.dropout(attn_weights, self.dropout_p, self.training)

            attn_output = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v)

            return rearrange(attn_output, "b s h d -> b s (h d)")

    def compute_q_kv(self, x: torch.Tensor, x_kv: Optional[torch.Tensor] = None) -> tuple:
        """
        Compute query, key, and value matrices from the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            x_kv (Optional[torch.Tensor]): Optional input for cross-attention of shape (batch_size, kv_seq_length, d_model).

        Returns:
            tuple: If self-attention, returns a combined qkv tensor of shape (batch_size, seq_length, 3, num_heads, head_dim),
                   If cross-attention, returns a tuple containing query tensor of shape (batch_size, seq_length, num_heads, head_dim),
                   and combined key-value tensor of shape (batch_size, kv_seq_length, 2, num_heads, head_dim).
        """
        if self.cross_attn:
            # Cross-attention: compute query from x and key-value from x_kv
            q = self.q_proj(x)
            kv = self.kv_proj(x_kv)
            q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
            kv = rearrange(kv, "b s (two h d) -> b s two h d", two=2, h=self.num_heads)
            return None, q, kv
        else:
            # Self-attention: compute query, key, value from x
            qkv = self.qkv_proj(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
            return qkv, None, None
