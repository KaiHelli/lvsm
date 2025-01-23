from torch import nn
from .multi_head_attention import MultiHeadAttention
from .norm import LayerNorm
from .activations import get_activation_fn, get_weight_init_fn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        num_heads,
        d_ff,
        dropout_p,
        *,
        activation="relu",
        bias=True,
        pre_norm=False,
        qk_norm=False,
        qk_exp_seq_len=None,
        sdpa_kernel="auto",
        min_num_context_views = 2,
         max_num_context_views=2,
    ):
        """
        Encoder block for the Transformer model.

        Args:
            d_model: The model dimensionality.
            d_k: The key and query dimensionality per head.
            d_v: the value dimensionality per head.
            num_heads: The number of attention heads.
            d_ff: The feedforward dimensionality.
            dropout_p: The dropout probability.
            activation: The activation to apply in the mlp.
            bias: Whether to include bias in the linear layers.
            pre_norm: Apply layer norm in pre- or post-layer fashion.
            qk_norm: Whether to apply qk_norm in the attention layers.
            qk_exp_seq_len: If qk_norm is true, this defines the expected sequence lengths to initialize the QK normalization scaling parameter g_0.
            sdpa_kernel: The kernel to use for the SDPA operation.
        """
        super().__init__()
        self.activation = activation
        self.pre_norm = pre_norm

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            num_heads=num_heads,
            dropout_p=dropout_p,
            cross_attn=False,
            bias=bias,
            qk_norm=qk_norm,
            qk_exp_seq_len=qk_exp_seq_len,
            sdpa_kernel=sdpa_kernel,
            min_num_context_views = min_num_context_views,
            max_num_context_views= max_num_context_views,     
        )

        # Feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation)(),
            nn.Dropout(dropout_p),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Layer normalization
        self.norm1 = LayerNorm(d_model, bias=bias)
        self.norm2 = LayerNorm(d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        """
        init_fn = get_weight_init_fn(self.activation)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init_fn(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, n_src, attn_mask=None):
        """
        Forward pass of the encoder block.
        """
        if self.pre_norm:
            # Pre-Layer Normalization variant
            # Layer normalization before attention and feedforward
            attn = self.self_attn(n_src, self.norm1(x), causal=False, attn_mask=attn_mask)
            x = x + self.dropout(attn)

            ff = self.mlp(self.norm2(x))
            x = x + self.dropout(ff)
        else:
            # Post-Layer Normalization variant (original)
            # Self-attention
            attn = self.self_attn(x, causal=False,n_src=n_src, attn_mask=attn_mask)
            x = self.norm1(x + self.dropout(attn))

            # Feedforward
            ff = self.mlp(x)
            x = self.norm2(x + self.dropout(ff))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, bias, **block_args):
        """
        Transformer encoder module.
        """
        super().__init__()

        # Stack of encoder blocks
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model, bias=bias, **block_args) for _ in range(num_layers)])

        # Layer normalization for the final output
        self.norm = LayerNorm(d_model, bias=bias)

    def forward(self, x, n_src, attn_mask=None):
        """
        Forward pass of the transformer encoder.
        """
        for layer in self.layers:
            x = layer(x,n_src=n_src, attn_mask=attn_mask)

        # Final layer normalization
        x = self.norm(x)

        return x
