import torch
import torch.nn as nn
import pytest
import sys
import os
from unittest.mock import patch

from model.transformer import Transformer


def setup_models(pre_norm):
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.0

    # Reference Transformer
    reference_model = nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="relu",
        batch_first=True,
        norm_first=pre_norm,
        bias=True,
    )

    # Our implementation of Transformer
    our_model = Transformer(
        d_model=d_model,
        d_k=d_model // nhead,
        d_v=d_model // nhead,
        num_heads=nhead,
        d_ff=dim_feedforward,
        dropout_p=dropout,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        bias=True,
        pre_norm=pre_norm,
        activation="relu",
    )

    return reference_model, our_model


def copy_weights(reference_model, our_model):
    """
    Copies weights from the reference PyTorch transformer to our implementation.
    """
    with torch.no_grad():
        d_model = our_model.encoder.layers[0].self_attn.d_model
        d_ff = our_model.encoder.layers[0].mlp[0].out_features

        our_model.encoder.norm.weight.data = reference_model.encoder.norm.weight.data.clone()
        our_model.encoder.norm.bias.data = reference_model.encoder.norm.bias.data.clone()

        # Copy encoder weights layer by layer
        for ref_layer, our_layer in zip(reference_model.encoder.layers, our_model.encoder.layers):
            our_layer.self_attn.qkv_proj.weight.data = ref_layer.self_attn.in_proj_weight.data.clone()
            our_layer.self_attn.qkv_proj.bias.data = ref_layer.self_attn.in_proj_bias.data.clone()

            our_layer.self_attn.o_proj.weight.data = ref_layer.self_attn.out_proj.weight.data.clone()
            our_layer.self_attn.o_proj.bias.data = ref_layer.self_attn.out_proj.bias.data.clone()

            # Norm
            our_layer.norm1.weight.data = ref_layer.norm1.weight.data.clone()
            our_layer.norm1.bias.data = ref_layer.norm1.bias.data.clone()

            our_layer.norm2.weight.data = ref_layer.norm2.weight.data.clone()
            our_layer.norm2.bias.data = ref_layer.norm2.bias.data.clone()

            # Feedforward
            our_layer.mlp[0].weight.data = ref_layer.linear1.weight.data.clone()
            our_layer.mlp[0].bias.data = ref_layer.linear1.bias.data.clone()

            our_layer.mlp[3].weight.data = ref_layer.linear2.weight.data.clone()
            our_layer.mlp[3].bias.data = ref_layer.linear2.bias.data.clone()

        our_model.decoder.norm.weight.data = reference_model.decoder.norm.weight.data.clone()
        our_model.decoder.norm.bias.data = reference_model.decoder.norm.bias.data.clone()

        # Copy decoder weights layer by layer
        for ref_layer, our_layer in zip(reference_model.decoder.layers, our_model.decoder.layers):
            # Self-attention
            our_layer.self_attn.qkv_proj.weight.data = ref_layer.self_attn.in_proj_weight.data.clone()
            our_layer.self_attn.qkv_proj.bias.data = ref_layer.self_attn.in_proj_bias.data.clone()

            our_layer.self_attn.o_proj.weight.data = ref_layer.self_attn.out_proj.weight.data.clone()
            our_layer.self_attn.o_proj.bias.data = ref_layer.self_attn.out_proj.bias.data.clone()

            # Cross-attention
            # Split `in_proj_weight` and `in_proj_bias` from reference attention to q, k, v
            in_proj_weight = ref_layer.multihead_attn.in_proj_weight
            in_proj_bias = ref_layer.multihead_attn.in_proj_bias

            q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)
            q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3, dim=0)

            our_layer.cross_attn.q_proj.weight.data = q_proj_weight.data.clone()
            our_layer.cross_attn.q_proj.bias.data = q_proj_bias.data.clone()

            our_layer.cross_attn.kv_proj.weight.data[:d_model] = k_proj_weight.data.clone()
            our_layer.cross_attn.kv_proj.weight.data[d_model:] = v_proj_weight.data.clone()

            our_layer.cross_attn.kv_proj.bias.data[:d_model] = k_proj_bias.data.clone()
            our_layer.cross_attn.kv_proj.bias.data[d_model:] = v_proj_bias.data.clone()

            our_layer.cross_attn.o_proj.weight.data = ref_layer.multihead_attn.out_proj.weight.data.clone()
            our_layer.cross_attn.o_proj.bias.data = ref_layer.multihead_attn.out_proj.bias.data.clone()

            # Norm
            our_layer.norm1.weight.data = ref_layer.norm1.weight.data.clone()
            our_layer.norm1.bias.data = ref_layer.norm1.bias.data.clone()

            our_layer.norm2.weight.data = ref_layer.norm2.weight.data.clone()
            our_layer.norm2.bias.data = ref_layer.norm2.bias.data.clone()

            our_layer.norm3.weight.data = ref_layer.norm3.weight.data.clone()
            our_layer.norm3.bias.data = ref_layer.norm3.bias.data.clone()

            # Feedforward
            our_layer.mlp[0].weight.data = ref_layer.linear1.weight.data.clone()
            our_layer.mlp[0].bias.data = ref_layer.linear1.bias.data.clone()

            our_layer.mlp[3].weight.data = ref_layer.linear2.weight.data.clone()
            our_layer.mlp[3].bias.data = ref_layer.linear2.bias.data.clone()


# Parametrize the PyTorch version for testing
@pytest.fixture(params=["1.13.0", None])
def torch_version(request):
    return request.param


@pytest.fixture(params=["cuda", "cpu"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")
    return request.param


@pytest.mark.parametrize("pre_norm", [True, False])
def test_transformer_equivalence(device, torch_version, pre_norm):
    with patch(
        "torch.__version__",
        new=torch_version if torch_version is not None else torch.__version__,
    ):

        reference_model, our_model = setup_models(pre_norm)
        reference_model.to(device)
        our_model.to(device)
        copy_weights(reference_model, our_model)

        n_batch = 7
        n_seq_src = 10
        n_seq_tgt = 20
        d_model = 512

        # Create some random input data
        src = torch.rand((n_batch, n_seq_src, d_model)).to(device)  # (sequence length, batch size, d_model)
        tgt = torch.rand((n_batch, n_seq_tgt, d_model)).to(device)

        reference_output = reference_model(
            src,
            tgt,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(n_seq_tgt, device=device),
            tgt_is_causal=True,
        )
        our_output = our_model(src, tgt)

        # Ensure the outputs are close enough
        assert torch.allclose(
            reference_output, our_output, atol=1e-5
        ), "The outputs of the reference and our implementation do not match!"


if __name__ == "__main__":
    pytest.main([__file__])
