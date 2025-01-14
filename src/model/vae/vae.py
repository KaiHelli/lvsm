from diffusers import AutoencoderKL
from dataclasses import dataclass
from huggingface_hub import whoami
from einops import rearrange
import torch
from functools import wraps

@dataclass
class VAECfg:
    hf_model_id: str
    hf_gated: bool
    num_latent_channels: int
    downsample_factor: int

class VAE(torch.nn.Module):
    @classmethod
    def from_cfg(cls, cfg: VAECfg) -> "VAE":
        return cls(hf_model_id=cfg.hf_model_id, hf_gated=cfg.hf_gated, num_latent_channels=cfg.num_latent_channels, downsample_factor=cfg.downsample_factor)

    def __init__(self, hf_model_id: str, hf_gated: bool, num_latent_channels: int, downsample_factor: int):
        super().__init__()

        if hf_gated:
            # Will raise LocalTokenNotFoundError if no token is found
            user = whoami()
            print(f"Hugging Face model {hf_model_id} is defined to be gated. Attempting to access the model with user [{user['fullname']}]...")

        self.vae = AutoencoderKL.from_pretrained(hf_model_id, use_safetensors=True, subfolder="vae")
        self.num_latent_channels = num_latent_channels
        self.downsample_factor = downsample_factor

        # Brief test if num_latent_channels and downsample_factor is valid by feeding a random tensor through the model
        test_input = torch.zeros(1, 3, downsample_factor, downsample_factor)
        test_output = self.encode(test_input)
        assert test_output.shape == (1, num_latent_channels, 1, 1), f"Expected output shape (1, {num_latent_channels}, 1, 1) but got {test_output.shape}"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample
    
    def get_latent_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.num_latent_channels, input_shape[0] // self.downsample_factor, input_shape[1] // self.downsample_factor)