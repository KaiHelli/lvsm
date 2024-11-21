import torch
import pytest
import sys
import os
from unittest.mock import patch
from packaging.version import parse as parse_version
from model.transformer.multi_head_attention import MultiHeadAttention


# Parametrize the device fixture to explicitly run on both 'cuda' and 'cpu'
@pytest.fixture(params=["cuda", "cpu"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")
    return request.param


# Parametrize the precision for testing: float32 and float16
@pytest.fixture(
    params=[torch.float16, torch.float32]
)  # rounding errors of bfloat16 seem to be to high for good comparison of outputs
def precision(request):
    return request.param


# Parametrize the PyTorch version for testing
@pytest.fixture(params=["1.13.0", None])
def torch_version(request):
    return request.param


@pytest.mark.parametrize("cross_attention", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_attention(device, precision, cross_attention, torch_version, causal):
    with (
        patch(
            "torch.__version__",
            new=torch_version if torch_version is not None else torch.__version__,
        ),
        torch.amp.autocast(device_type=device, dtype=precision),
    ):
        print(
            f"Testing attention on {device = }, {precision = }, {cross_attention = }, {torch_version = }, {causal = }."
        )
        n_batch = 7
        nhead = 4
        n_seq_q = 534
        n_seq_kv = 316
        embed_dim = 128

        # Prepare input tensors
        x = torch.normal(torch.tensor(0.0), torch.tensor(1.0), size=(n_batch, n_seq_q, embed_dim)).to(device)
        x_q = torch.normal(torch.tensor(0.0), torch.tensor(1.0), size=(n_batch, n_seq_q, embed_dim)).to(device)
        x_kv = torch.normal(torch.tensor(0.0), torch.tensor(1.0), size=(n_batch, n_seq_kv, embed_dim)).to(device)

        # Create attention mask
        if cross_attention:
            attn_mask = torch.randint(0, 2, (n_seq_q, n_seq_kv), dtype=torch.bool, device=device)
            causal_mask = torch.tril(torch.ones((n_seq_q, n_seq_kv), dtype=torch.bool, device=device))
        else:
            attn_mask = torch.randint(0, 2, (n_seq_q, n_seq_q), dtype=torch.bool, device=device)
            causal_mask = torch.tril(torch.ones((n_seq_q, n_seq_q), dtype=torch.bool, device=device))

        att_ref = torch.nn.MultiheadAttention(embed_dim, nhead, batch_first=True, bias=True, device=device)
        att_test = MultiHeadAttention(
            d_model=embed_dim,
            d_k=embed_dim // nhead,
            d_v=embed_dim // nhead,
            num_heads=nhead,
            dropout_p=0.0,
            cross_attn=cross_attention,
            bias=True,
        )

        # Load state_dict from reference attention to the custom attention
        with torch.no_grad():
            # Split `in_proj_weight` and `in_proj_bias` from reference attention to q, k, v
            in_proj_weight = att_ref.in_proj_weight
            in_proj_bias = att_ref.in_proj_bias

            q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
            q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3, dim=0)

            if cross_attention:
                # Assign weights to the `q_proj` and `kv_proj` layers of custom attention
                att_test.q_proj.weight.data = q_proj_weight
                att_test.kv_proj.weight.data[:embed_dim] = k_proj_weight
                att_test.kv_proj.weight.data[embed_dim:] = v_proj_weight

                # Assign biases to the `q_proj` and `kv_proj` layers of custom attention
                att_test.q_proj.bias.data = q_proj_bias
                att_test.kv_proj.bias.data[:embed_dim] = k_proj_bias
                att_test.kv_proj.bias.data[embed_dim:] = v_proj_bias
            else:
                # Assign weights and biases to the `qkv_proj` layer for self-attention
                att_test.qkv_proj.weight.data[:embed_dim] = q_proj_weight
                att_test.qkv_proj.weight.data[embed_dim : 2 * embed_dim] = k_proj_weight
                att_test.qkv_proj.weight.data[2 * embed_dim :] = v_proj_weight

                att_test.qkv_proj.bias.data[:embed_dim] = q_proj_bias
                att_test.qkv_proj.bias.data[embed_dim : 2 * embed_dim] = k_proj_bias
                att_test.qkv_proj.bias.data[2 * embed_dim :] = v_proj_bias

            # Copy output projection weights and biases
            att_test.o_proj.weight.data = att_ref.out_proj.weight.data
            att_test.o_proj.bias.data = att_ref.out_proj.bias.data

        att_ref = att_ref.to(device)
        att_test = att_test.to(device)

        # Perform the test comparison
        if cross_attention:
            y, _ = att_ref(
                x_q,
                x_kv,
                x_kv,
                attn_mask=(~causal_mask if causal else ~attn_mask if attn_mask is not None else None),
                is_causal=causal,
            )
            y_ = att_test(x_q, x_kv, attn_mask=None if causal else attn_mask, causal=causal)
        else:
            y, _ = att_ref(
                x,
                x,
                x,
                attn_mask=(~causal_mask if causal else ~attn_mask if attn_mask is not None else None),
                is_causal=causal,
            )
            y_ = att_test(x, attn_mask=None if causal else attn_mask, causal=causal)

        assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5

        # Test add_input functionality
        if cross_attention:
            x_q_ = x_q.clone()
            y__ = att_test(x_q_, x_kv, attn_mask=None if causal else attn_mask, causal=causal) + x_q
            assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5
        else:
            x_ = x.clone()
            y__ = att_test(x_, attn_mask=None if causal else attn_mask, causal=causal) + x
            assert torch.sqrt(torch.nn.functional.mse_loss(y + x, y__)) < 5e-5

        # Test with no gradient update
        if cross_attention:
            x_q_ = x_q.clone()
            with torch.no_grad():
                y__ = att_test(x_q_, x_kv, attn_mask=None if causal else attn_mask, causal=causal) + x_q
            assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5
        else:
            x_ = x.clone()
            with torch.no_grad():
                y__ = att_test(x_, attn_mask=None if causal else attn_mask, causal=causal) + x
            assert torch.sqrt(torch.nn.functional.mse_loss(y + x, y__)) < 5e-5


if __name__ == "__main__":
    pytest.main([__file__])
