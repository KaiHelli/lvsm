import torch
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from ..dataset.types import DataShim


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        num_heads,
        d_ff,
        dropout_p,
        num_encoder_layers=0,
        num_decoder_layers=0,
        bias=True,
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
            bias: Whether to include bias in the attention layers.
        """
        super().__init__()

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
            )
        else:
            self.decoder = None

    def forward(self, src=None, tgt=None, src_mask=None, tgt_mask=None):
        """
        Forward pass for the transformer model.

        Args:
            src: Source sequence tensor (for encoder input).
            tgt: Target sequence tensor (for decoder input).
            src_mask: Mask for the source sequence (encoder).
            tgt_mask: Mask for the target sequence (decoder).
        """
        if self.encoder is not None and src is not None:
            enc_output = self.encoder(src, attn_mask=src_mask)
        else:
            enc_output = None

        if self.decoder is not None and tgt is not None:
            if enc_output is None:
                raise ValueError("Decoder-only transformer requires non-null target input (tgt).")
            dec_output = self.decoder(tgt, enc_output, self_attn_mask=tgt_mask, cross_attn_mask=src_mask)
            return dec_output

        if self.encoder is not None and self.decoder is None:
            return enc_output

        if self.decoder is not None and self.encoder is None:
            dec_output = self.decoder(tgt, None, self_attn_mask=tgt_mask)
            return dec_output

        raise ValueError("At least one of encoder or decoder must be specified.")

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
