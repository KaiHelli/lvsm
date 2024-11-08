from torch import nn
from .multi_head_attention import MultiHeadAttention
from .norm import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, d_ff, dropout_p, bias=True):
        """
        Decoder block for the Transformer model.

        Args:
            d_model: The model dimensionality.
            num_heads: The number of attention heads.
            d_ff: The feedforward dimensionality.
            dropout_p: The dropout probability.
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            num_heads=num_heads,
            dropout_p=dropout_p,
            cross_attn=False,
            bias=bias,
        )

        # Multi-head cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            num_heads=num_heads,
            dropout_p=dropout_p,
            cross_attn=True,
            bias=bias,
        )

        # Feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        """
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x, enc_output, causal=True, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass of the decoder block.

        Args:
            x: Input tensor (target sequence representation).
            enc_output: Output tensor from the encoder (source sequence representation).
            self_attn_mask: Mask for the self-attention mechanism.
            cross_attn_mask: Mask for the encoder-decoder attention mechanism.
            bias: Whether to include bias in the attention layers.
        """
        # Self-attention
        attn = self.self_attn(x, causal=causal, attn_mask=self_attn_mask)
        x = x + self.dropout(attn)

        # Layer normalization
        x = self.norm1(x)

        # Cross-attention (encoder-decoder attention)
        cross_attn = self.cross_attn(x, enc_output, causal=False, attn_mask=cross_attn_mask)
        x = x + self.dropout(cross_attn)

        # Layer normalization
        x = self.norm2(x)

        # Feedforward
        ff = self.mlp(x)
        x = x + self.dropout(ff)

        # Layer normalization
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """
        Transformer decoder module.

        Args:
            num_layers: Number of decoder layers.
            **block_args: Arguments for the decoder block.
        """
        super().__init__()

        # Stack of decoder blocks
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, enc_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Forward pass of the transformer decoder.

        Args:
            x: Input tensor (target sequence representation).
            enc_output: Output tensor from the encoder (source sequence representation).
            self_attn_mask: Mask for the self-attention mechanism.
            cross_attn_mask: Mask for the encoder-decoder attention mechanism.
        """
        for layer in self.layers:
            x = layer(
                x,
                enc_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
        return x
