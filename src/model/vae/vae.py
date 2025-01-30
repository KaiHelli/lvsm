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
    stochastic_encoder: bool = True


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
            stochastic_encoder=cfg.stochastic_encoder,
        )

    def __init__(
        self,
        hf_model_id: str,
        hf_subfolder: str,
        hf_gated: bool,
        num_latent_channels: int,
        downsample_factor: int,
        num_slices_in_parallel: int | None = None,
        stochastic_encoder: bool = True
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
        self.stochastic_encoder = stochastic_encoder

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
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode an input tensor into VAE latents.
        If `x` is 4D (B, C, H, W), encode directly.
        If `x` is 5D ((B1 B2), N, C, H, W), slice along the batch dimension for parallel processing.
        """
        # ----- 4D case -----
        if x.ndim == 4:
            latent_dist = self.vae.encode(x, return_dict=False)[0]
            mean, std = latent_dist.mean, latent_dist.std
            return self.sample(mean, std) if sample else (mean, std)

        # ----- 5D case -----
        b = x.shape[0]
        # If num_slices_in_parallel is None, process everything at once (chunk_size = b)
        chunk_size = b if self.num_slices_in_parallel is None else self.num_slices_in_parallel

        # Split along the batch dimension into chunks of size `chunk_size`
        x_chunks = torch.split(x, split_size_or_sections=chunk_size, dim=0)

        mean_chunks, std_chunks = [], []
        for chunk in x_chunks:
            # chunk: [chunk_b, n, c, h, w]
            chunk_b = chunk.shape[0]

            # Flatten the first two dimensions for VAE: [chunk_b*n, c, h, w]
            chunk = rearrange(chunk, "b n c h w -> (b n) c h w")

            latent_dist = self.vae.encode(chunk, return_dict=False)[0]
            chunk_mean, chunk_std = latent_dist.mean, latent_dist.std

            # Reshape back to 5D: [chunk_b, n, c, h, w]
            chunk_mean = rearrange(chunk_mean, "(b n) c h w -> b n c h w", b=chunk_b)
            chunk_std = rearrange(chunk_std, "(b n) c h w -> b n c h w", b=chunk_b)

            mean_chunks.append(chunk_mean)
            std_chunks.append(chunk_std)

        mean = torch.cat(mean_chunks, dim=0)
        std = torch.cat(std_chunks, dim=0)

        return self.sample(mean, std) if sample else (mean, std)

        # The following causes issues with torch.compile(), so we do it manually above
        # return self.vae.encode(x).latent_dist.sample()

    def decode(self, z: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Decode latents back to image space.
        If `z` is 4D (B, C, H, W), decode directly.
        If `z` is 5D ((B1 B2), N, C, H, W), slice along
        the batch dimension for parallel processing.
        """
        # Allow outer scope to take precedence over requires_grad using e.g. torch.no_grad() contexts
        # Useful e.g. for evaluation where gradients are not needed
        requires_grad = torch.is_grad_enabled() and requires_grad

        with torch.set_grad_enabled(requires_grad):
            # ----- 4D case -----
            if z.ndim == 4:
                return self.vae.decode(z, return_dict=False)[0]

            # ----- 5D case -----
            b = z.shape[0]
            # If num_slices_in_parallel is None, process everything at once (chunk_size = b)
            chunk_size = b if self.num_slices_in_parallel is None else self.num_slices_in_parallel

            # Split along the batch dimension into chunks of size `chunk_size`
            z_chunks = torch.split(z, split_size_or_sections=chunk_size, dim=0)

            decoded_chunks = []
            for chunk in z_chunks:
                # chunk: [chunk_b, n, c, h, w]
                chunk_b = chunk.shape[0]

                # Flatten the first two dimensions for VAE: [chunk_b*n, c, h, w]
                chunk = rearrange(chunk, "b n c h w -> (b n) c h w")

                decoded = self.vae.decode(chunk, return_dict=False)[0]

                # Reshape back to 5D: [chunk_b, n, c, h, w]
                decoded = rearrange(decoded, "(b n) c h w -> b n c h w", b=chunk_b)
                decoded_chunks.append(decoded)

            out = torch.cat(decoded_chunks, dim=0)
            return out

    def sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.stochastic_encoder:
            return mean + std * torch.randn_like(mean, device=mean.device)
        else:
            return mean

    def get_latent_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (
            self.num_latent_channels,
            input_shape[0] // self.downsample_factor,
            input_shape[1] // self.downsample_factor,
        )
