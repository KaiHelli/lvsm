import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass, asdict
from ..transformer import Transformer, TransformerCfg
from ..transformer.norm import LayerNorm
from src.dataset.types import DataShim
from src.dataset.shims.plucker_rays import generate_rays_batch


@dataclass
class LVSMCfg:
    patch_size: int
    num_channels: int
    transformer_cfg: TransformerCfg


class LVSM(torch.nn.Module):
    @classmethod
    def from_cfg(cls, cfg: LVSMCfg) -> "LVSM":
        return cls(transformer_cfg=cfg.transformer_cfg, patch_size=cfg.patch_size, num_channels=cfg.num_channels)

    def __init__(
        self,
        *,
        transformer_cfg: TransformerCfg,
        patch_size: int,
        num_channels: int,
    ):
        """
        Transformer model that can be used as encoder-decoder, encoder-only, or decoder-only.
        """
        super().__init__()

        self.patch_size = patch_size
        self.num_channels = num_channels

        assert transformer_cfg.num_decoder_layers == 0, "Only decoder-only models are supported."

        self.transformer = Transformer.from_config(cfg=transformer_cfg)

        # TODO: set bias as false in those linear layers as well?
        self.tokenize_input = torch.nn.Sequential(
            Rearrange("b n c (h ph) (w pw) -> b n h w (c ph pw)", ph=patch_size, pw=patch_size),
            torch.nn.Linear((num_channels + 6) * (patch_size**2), transformer_cfg.d_model),
        )

        self.tokenize_target = torch.nn.Sequential(
            Rearrange("b n c (h ph) (w pw) -> b n h w (c ph pw)", ph=patch_size, pw=patch_size),
            torch.nn.Linear(6 * (patch_size**2), transformer_cfg.d_model),
        )

        self.untokenize_output = torch.nn.Sequential(
            torch.nn.Linear(transformer_cfg.d_model, num_channels * (patch_size**2)),
            torch.nn.Sigmoid(),
            Rearrange("b n h w (c ph pw) -> b n c (h ph) (w pw)", ph=patch_size, pw=patch_size, c=num_channels),
        )

    def forward(self, src_img, src_rays, tgt_rays):
        # Tokenize input images and rays
        tkn_src = self.tokenize_input(torch.cat([src_img, src_rays], dim=-3))

        # Tokenize target rays
        rays_tkn_tgt = self.tokenize_target(tgt_rays)

        tgt_seq_shape = rays_tkn_tgt.shape[1:4]

        # Flatten the sequence dimension
        tkn_src = rearrange(tkn_src, "b n h w d -> b (n h w) d")
        tkn_tgt = rearrange(rays_tkn_tgt, "b n h w d -> b (n h w) d")

        # This is a decoder-only model. Concatenate the source and target tokens.
        tkn_in = torch.cat([tkn_src, tkn_tgt], dim=1)

        # num target views, num patches in height, num patches in width
        n, h, w = tgt_seq_shape

        # num_tgt_tokens = n * h * w
        num_tgt_tokens = tkn_tgt.shape[1]
        # num_src_tokens can differ due to the number of input images
        num_src_tokens = tkn_src.shape[1]
        total_tokens = tkn_in.shape[1]

        ## Create attention mask
        # Step 1: Initialize the final attention mask with zeros
        # Create an attention mask of shape [(total number of tokens), (total number of tokens)].
        final_attn_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=tkn_src.device)

        # Step 2: Fill in intra-target attention (block diagonal for each target view)
        # Target tokens can fully attend to tokens within the same target view but not to tokens in other target views.
        ones_block = torch.ones((h * w, h * w), dtype=torch.bool, device=tkn_src.device)
        intra_target_mask = torch.block_diag(*[ones_block for _ in range(n)])
        final_attn_mask[num_src_tokens:, num_src_tokens:] = intra_target_mask

        # Step 3: Fill in target-to-source and source-to-source attention.
        # Target tokens attend to all source tokens for contextual information.
        final_attn_mask[:, : tkn_src.shape[1]] = True

        # Example with 1 source view (2 tokens) and 2 target views (each with 2 tokens):
        # [ 1 1 | 0 0 0 0 ]  <- Source tokens do only attend to themselves.
        # [ 1 1 | 0 0 0 0 ]
        # -------------------
        # [ 1 1 | 1 1 0 0 ]  <- Target tokens from view 1 attend to themselves and source tokens.
        # [ 1 1 | 1 1 0 0 ]
        # -------------------
        # [ 1 1 | 0 0 1 1 ]  <- Target tokens from view 2 attend to themselves and source tokens.
        # [ 1 1 | 0 0 1 1 ]
        #
        # - The upper left block (zero_mask) indicates that source tokens do not attend to any tokens.
        # - The lower left blocks (tgt_src_attn_mask) indicate that target tokens attend to all source tokens.
        # - The lower right blocks (tgt_attn_mask) represent intra-target attention, where tokens in the same group can attend to each other.

        tkn_out = self.transformer(src=tkn_in, src_mask=final_attn_mask)

        # Separate the target tokens from the source tokens
        tkn_tgt_out = tkn_out[:, num_src_tokens:]

        # Unflatten the sequence dimension
        tkn_tgt_out = rearrange(
            tkn_tgt_out, "b (n h w) d -> b n h w d", n=tgt_seq_shape[0], h=tgt_seq_shape[1], w=tgt_seq_shape[2]
        )

        # Untokenize the output
        tgt_img = self.untokenize_output(tkn_tgt_out)

        return tgt_img

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return generate_rays_batch
