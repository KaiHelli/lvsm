from torch import nn
from .multi_head_attention import MultiHeadAttention
from .norm import LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, d_ff, dropout_p, bias=True):
        """
        Encoder block for the Transformer model.

        Args:
            d_model: The model dimensionality.
            num_heads: The number of attention heads.
            d_ff: The feedforward dimensionality.
            dropout_p: The dropout probability.
            bias: Whether to include bias in the attention layers.
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

    def forward(self, x, attn_mask=None):
        """
        Forward pass of the encoder block.
        """
        # Self-attention
        attn = self.self_attn(x, causal=False, attn_mask=attn_mask)
        x = x + self.dropout(attn)

        # Layer normalization
        x = self.norm1(x)

        # Feedforward
        ff = self.mlp(x)
        x = x + self.dropout(ff)

        # Layer normalization
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """
        Transformer encoder module.
        """
        super().__init__()

        # Stack of encoder blocks
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, attn_mask=None):
        """
        Forward pass of the transformer encoder.
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
