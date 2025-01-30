import torch
from typing import List
from einops import rearrange
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from math import log2

from src.dataset.types import DataShim
from src.dataset.shims.plucker_rays import generate_rays_batch
from src.dataset.shims.relative_poses import encode_relative_poses_batch_avgc

from ..transformer import Transformer, TransformerCfg
from ..vae import VAE, VAECfg
from ..transformer.norm import LayerNorm, LayerNorm2d
from ..transformer.activations import get_activation_fn, get_weight_init_fn


@dataclass
class LVSMCfg:
    patch_size: int
    num_input_channels: int
    transformer_cfg: TransformerCfg
    vae_cfg: VAECfg | None


class LVSM(torch.nn.Module):
    @classmethod
    def from_cfg(cls, cfg: LVSMCfg) -> "LVSM":
        return cls(
            transformer_cfg=cfg.transformer_cfg,
            vae_cfg=cfg.vae_cfg,
            patch_size=cfg.patch_size,
            num_input_channels=cfg.num_input_channels,
        )

    def __init__(
        self,
        *,
        transformer_cfg: TransformerCfg,
        vae_cfg: VAECfg | None,
        patch_size: int,
        num_input_channels: int,
    ):
        """
        Transformer model that can be used as encoder-decoder, encoder-only, or decoder-only.
        """
        super().__init__()

        self.patch_size = patch_size
        self.num_input_channels = num_input_channels

        assert transformer_cfg.num_decoder_layers == 0, "Only decoder-only models are supported."

        self.transformer = Transformer.from_config(cfg=transformer_cfg)

        if vae_cfg is None:
            self.vae = None
            self.num_patch_channels = self.num_input_channels
            num_ray_channels = 6
        else:
            self.vae = VAE.from_cfg(vae_cfg)

            self.num_patch_channels = self.vae.num_latent_channels
            num_ray_channels = self.num_patch_channels

            # Use a simple CNN to downsample the rays to match the latent shape of the VAE
            # e.g. downsample by self.vae.downsample_factor
            # TODO: Make this configurable and factor out the downsample CNN
            downsample_factor = self.vae.downsample_factor
            assert downsample_factor & (downsample_factor - 1) == 0, "Downsample factor must be a power of 2."

            num_layers = int(log2(downsample_factor))

            layers = []
            channels = torch.linspace(6, 16, num_layers + 1, dtype=torch.int).tolist()
            for i in range(num_layers):
                conv = torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=2, stride=2, bias=transformer_cfg.bias)
                norm = LayerNorm2d(channels[i + 1])
                act = get_activation_fn("gelu")()

                layers += [conv, norm, act]

            layers += [torch.nn.Conv2d(channels[-1], self.num_patch_channels, kernel_size=1, bias=transformer_cfg.bias)]

            self.downsample_rays = torch.nn.Sequential(*layers)
            init_fn = get_weight_init_fn("gelu")

            # Initialize the weights
            for layer in self.downsample_rays:
                if isinstance(layer, torch.nn.Conv2d):
                    init_fn(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        self.tokenize_input = torch.nn.Sequential(
            Rearrange("b n c (h ph) (w pw) -> b n h w (c ph pw)", ph=patch_size, pw=patch_size),
            torch.nn.Linear(
                (self.num_patch_channels + num_ray_channels) * (patch_size**2),
                transformer_cfg.d_model,
                bias=transformer_cfg.bias,
            ),
        )

        self.tokenize_target = torch.nn.Sequential(
            Rearrange("b n c (h ph) (w pw) -> b n h w (c ph pw)", ph=patch_size, pw=patch_size),
            torch.nn.Linear(num_ray_channels * (patch_size**2), transformer_cfg.d_model, bias=transformer_cfg.bias),
        )

        self.untokenize_output = torch.nn.Sequential(
            torch.nn.Linear(
                transformer_cfg.d_model, self.num_patch_channels * (patch_size**2), bias=transformer_cfg.bias
            ),
            Rearrange(
                "b n h w (c ph pw) -> b n c (h ph) (w pw)", ph=patch_size, pw=patch_size, c=self.num_patch_channels
            ),
        )

        self.norm_in = LayerNorm(transformer_cfg.d_model, bias=transformer_cfg.bias)
        self.norm_out = LayerNorm(transformer_cfg.d_model, bias=transformer_cfg.bias)

    def forward(self, src_img, src_rays, tgt_rays, attn_mask, vae_latents=None, decode_latents=True):
        # If a VAE is used, encode the source image and downsample the rays
        if self.vae is not None:
            b = src_img.shape[0]
            # Reshape the source image to match the expected input shape
            src_rays, tgt_rays = [rearrange(x, f"b n c h w -> (b n) c h w") for x in [src_rays, tgt_rays]]

            if vae_latents is not None:
                src_img = self.vae.sample(vae_latents["mean"], vae_latents["std"])
            else:
                # Encode source image
                src_img = self.vae.encode(src_img)

            # Downsample source rays
            src_rays = self.downsample_rays(src_rays)
            # Downsample target rays
            tgt_rays = self.downsample_rays(tgt_rays)

            # Reshape the source image back to the original shape
            src_rays, tgt_rays = [rearrange(x, f"(b n) c h w -> b n c h w", b=b) for x in [src_rays, tgt_rays]]

        # Tokenize input images and rays
        tkn_src = self.tokenize_input(torch.cat([src_img, src_rays], dim=-3))

        # Tokenize target rays
        rays_tkn_tgt = self.tokenize_target(tgt_rays)

        # num_src_views = tkn_src.shape[1]
        num_tgt_views, num_patches_h, num_patches_w = rays_tkn_tgt.shape[1:4]
        # num_tkn_per_view = num_patches_h * num_patches_w

        # Flatten the sequence dimension
        tkn_src = rearrange(tkn_src, "b n h w d -> b (n h w) d")
        tkn_tgt = rearrange(rays_tkn_tgt, "b n h w d -> b (n h w) d")

        # This is a decoder-only model. Concatenate the source and target tokens.
        tkn_in = torch.cat([tkn_src, tkn_tgt], dim=1)

        # Normalize
        tkn_in = self.norm_in(tkn_in)

        # num_tgt_tokens = n * h * w
        # num_tgt_tokens = tkn_tgt.shape[1]
        # num_src_tokens can differ due to the number of input images
        num_src_tokens = tkn_src.shape[1]
        # total_tokens = tkn_in.shape[1]

        tkn_out = self.transformer(src=tkn_in, src_mask=attn_mask)

        # Separate the target tokens from the source tokens
        tkn_tgt_out = tkn_out[:, num_src_tokens:]

        # Unflatten the sequence dimension
        tkn_tgt_out = rearrange(
            tkn_tgt_out, "b (n h w) d -> b n h w d", n=num_tgt_views, h=num_patches_h, w=num_patches_w
        )

        # Normalize
        tkn_tgt_out = self.norm_out(tkn_tgt_out)

        # Untokenize the output
        tgt_img = self.untokenize_output(tkn_tgt_out)

        # If a VAE is used, decode the output
        if self.vae is not None:
            tgt_latent = tgt_img
            tgt_img = None
            if decode_latents:
                tgt_img = self.vae.decode(tgt_latent).clamp(0, 1)
        else:
            tgt_latent = None
            # Map the output to [0, 1] for RGB output
            tgt_img = torch.nn.functional.sigmoid(tgt_img)

        return tgt_img, tgt_latent

    def get_data_shim(self) -> List[DataShim]:
        """The default shim doesn't modify the batch."""
        return [encode_relative_poses_batch_avgc, generate_rays_batch]

    def get_num_tkn_per_view(self, img_height: int, img_width: int) -> int:
        downsample_factor = self.vae.downsample_factor if self.vae is not None else 1

        return ((img_height // downsample_factor) * (img_width // downsample_factor)) // (self.patch_size**2)
