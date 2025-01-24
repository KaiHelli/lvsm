import torch
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from src.dataset.types import DataShim
from dataclasses import dataclass
from typing import Optional, List, Literal


@dataclass
class TransformerCfg:
    d_model: int
    d_k: int
    d_v: int
    num_heads: int
    d_ff: int
    dropout_p: float
    num_encoder_layers: int
    num_decoder_layers: int
    bias: bool
    activation: str
    pre_norm: bool
    qk_norm: int
    qk_exp_seq_len: Optional[int]
    sdpa_kernel: Literal["flex-attention", "torch-sdpa", "naive", "auto"]


class Transformer(torch.nn.Module):
    @classmethod
    def from_config(cls, cfg: TransformerCfg) -> "Transformer":
        return cls(
            d_model=cfg.d_model,
            d_k=cfg.d_k,
            d_v=cfg.d_v,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            dropout_p=cfg.dropout_p,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            activation=cfg.activation,
            bias=cfg.bias,
            pre_norm=cfg.pre_norm,
            qk_norm=cfg.qk_norm,
            qk_exp_seq_len=cfg.qk_exp_seq_len,
            sdpa_kernel=cfg.sdpa_kernel,
        )

    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        num_heads,
        d_ff,
        dropout_p,
        *,
        num_encoder_layers=0,
        num_decoder_layers=0,
        activation="relu",
        bias=True,
        pre_norm=False,
        qk_norm=False,
        qk_exp_seq_len=None,
        sdpa_kernel="auto",
    ):
        """
        Transformer model that can be used as encoder-decoder, encoder-only, or decoder-only.

        Args:
            d_model: Model dimensionality.
            d_k: Key dimensionality.
            d_v: Value dimensionality.
            num_heads: Number of attention heads.
            d_ff: Feedforward network dimensionality.
            dropout_p: Dropout probability.
            num_encoder_layers: Number of encoder layers (0 if not using an encoder).
            num_decoder_layers: Number of decoder layers (0 if not using a decoder).
            activation: The activation to apply in the mlps.
            bias: Whether to include bias in the attention layers.
            norm: The normalization strategy to apply.
            qk_norm: Whether to apply qk_norm in the attention layers.
            qk_exp_seq_len: If qk_norm is true, this defines the expected sequence lengths to initialize the QK normalization scaling parameter g_0.
        """
        super().__init__()

        assert num_encoder_layers + num_decoder_layers > 0, "At least one of encoder or decoder must be specified."

        # Encoder module if required
        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_layers=num_encoder_layers,
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                bias=bias,
                activation=activation,
                pre_norm=pre_norm,
                qk_norm=qk_norm,
                qk_exp_seq_len=qk_exp_seq_len,
                sdpa_kernel=sdpa_kernel,
            )
        else:
            self.encoder = None

        # Decoder module if required
        if num_decoder_layers > 0:
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                bias=bias,
                activation=activation,
                pre_norm=pre_norm,
                qk_norm=qk_norm,
                qk_exp_seq_len=qk_exp_seq_len,
                sdpa_kernel=sdpa_kernel,
            )
        else:
            self.decoder = None

    def forward(self, src=None, tgt=None, *, src_mask=None, tgt_causal=True, tgt_sa_mask=None, tgt_ca_mask=None):
        """
        Forward pass for the transformer model.

        Args:
            src: Source sequence tensor (for encoder input).
            tgt: Target sequence tensor (for decoder input).
            src_mask: Mask for the source sequence (encoder).
            tgt_mask: Mask for the target sequence (decoder).
        """
        if self.encoder is not None and self.decoder is not None:
            assert src is not None and tgt is not None, "Transformer requires non-null input for src and tgt."
            enc_output = self.encoder(src, attn_mask=src_mask)

            dec_output = self.decoder(
                tgt, enc_output, causal=tgt_causal, self_attn_mask=tgt_sa_mask, cross_attn_mask=tgt_ca_mask
            )

            return dec_output
        elif self.encoder is not None:
            assert src is not None, "Transformer requires non-null input for src."

            enc_output = self.encoder(src, attn_mask=src_mask)

            return enc_output
        elif self.decoder is not None:
            assert tgt is not None, "Transformer requires non-null input for tgt."

            dec_output = self.decoder(tgt, None, causal=tgt_causal, self_attn_mask=tgt_sa_mask)

            return dec_output

    def get_data_shim(self) -> List[DataShim]:
        return []
