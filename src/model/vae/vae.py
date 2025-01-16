from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from dataclasses import dataclass
from huggingface_hub import whoami
from einops import rearrange
import torch


@dataclass
class VAECfg:
    hf_model_id: str
    hf_subfolder: str
    hf_gated: bool
    num_latent_channels: int
    downsample_factor: int
    num_slices_in_parallel: int | None = None


class VAE(torch.nn.Module):
    @classmethod
    def from_cfg(cls, cfg: VAECfg) -> "VAE":
        return cls(
            hf_model_id=cfg.hf_model_id,
            hf_subfolder=cfg.hf_subfolder,
            hf_gated=cfg.hf_gated,
            num_latent_channels=cfg.num_latent_channels,
            downsample_factor=cfg.downsample_factor,
            num_slices_in_parallel=cfg.num_slices_in_parallel,
        )

    def __init__(
        self,
        hf_model_id: str,
        hf_subfolder: str,
        hf_gated: bool,
        num_latent_channels: int,
        downsample_factor: int,
        num_slices_in_parallel: int | None = None,
    ):
        super().__init__()

        if hf_gated:
            # Will raise LocalTokenNotFoundError if no token is found
            user = whoami()
            print(
                f"Hugging Face model {hf_model_id} is defined to be gated. Attempting to access the model with user [{user['fullname']}]..."
            )

        self.vae = AutoencoderKL.from_pretrained(
            hf_model_id, use_safetensors=True, subfolder=hf_subfolder, torch_dtype=torch.bfloat16
        )

        # Freeze the VAE and set it to eval mode
        self.vae.requires_grad_(False)
        self.vae.eval()

        # self.vae.enable_tiling(use_tiling=True)
        # self.vae.enable_slicing()

        self.num_latent_channels = num_latent_channels
        self.downsample_factor = downsample_factor
        self.num_slices_in_parallel = num_slices_in_parallel

        # Brief test if num_latent_channels and downsample_factor is valid by feeding a zero tensor through the model
        test_input = torch.zeros(
            1, 3, downsample_factor, downsample_factor, dtype=torch.bfloat16, device=self.vae.device
        )
        test_output = self.encode(test_input)
        assert test_output.shape == (
            1,
            num_latent_channels,
            1,
            1,
        ), f"Expected output shape (1, {num_latent_channels}, 1, 1) but got {test_output.shape}"

        del test_input, test_output

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input tensor into VAE latents.
        If `x` is 4D (B, C, H, W), encode directly.
        If `x` is 5D ((B1 B2), N, C, H, W), slice along the batch dimension for parallel processing.
        """
        # ----- 4D case -----
        if x.ndim == 4:
            latent_dist = self.vae.encode(x, return_dict=False)[0]
            mean, std = latent_dist.mean, latent_dist.std
            return mean + std * torch.randn_like(mean, device=mean.device)

        # ----- 5D case -----
        # Decide how many slices to process in parallel
        b = x.shape[0]
        num_slices = 1 if b == 1 else (self.num_slices_in_parallel or b)

        # Reshape to process slices in parallel
        x_slices = rearrange(x, "(b1 b2) n c h w -> b1 (b2 n) c h w", b2=num_slices)

        mean, std = [], []
        for x_slice in x_slices:
            latent_dist = self.vae.encode(x_slice, return_dict=False)[0]
            mean.append(latent_dist.mean)
            std.append(latent_dist.std)

        # Combine results from all slices
        mean = torch.stack(mean)
        std = torch.stack(std)

        # Reshape back to the original batch structure
        mean = rearrange(mean, "b1 (b2 n) c h w -> (b1 b2) n c h w", b2=num_slices)
        std = rearrange(std, "b1 (b2 n) c h w -> (b1 b2) n c h w", b2=num_slices)

        return mean + std * torch.randn_like(mean, device=mean.device)

        # The following causes issues with torch.compile(), so we do it manually above
        # return self.vae.encode(x).latent_dist.sample()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to image space.
        If `z` is 4D (B, C, H, W), decode directly.
        If `z` is 5D ((B1 B2), N, C, H, W), slice along
        the batch dimension for parallel processing.
        """
        # ----- 4D case -----
        if z.ndim == 4:
            return self.vae.decode(z, return_dict=False)[0]

        # ----- 5D case -----
        b = z.shape[0]
        num_slices = 1 if b == 1 else (self.num_slices_in_parallel or b)

        # Reshape to process slices in parallel
        z_slices = rearrange(z, "(b1 b2) n c h w -> b1 (b2 n) c h w", b2=num_slices)

        decoded_slices = [self.vae.decode(z_slice, return_dict=False)[0] for z_slice in z_slices]

        out = torch.stack(decoded_slices)
        out = rearrange(out, "b1 (b2 n) c h w -> (b1 b2) n c h w", b2=num_slices)

        return out

    def get_latent_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (
            self.num_latent_channels,
            input_shape[0] // self.downsample_factor,
            input_shape[1] // self.downsample_factor,
        )
