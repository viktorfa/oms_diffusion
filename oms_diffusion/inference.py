import os
import torch
from pathlib import Path
from diffusers import AutoencoderKL
from PIL import Image


DEFAULT_HG_ROOT = Path(os.getcwd()) / "oms_models"


class AbstractInferenceModel:
    def __init__(
        self,
        hg_root: str = None,
        cache_dir: str = None,
        device: str = "cuda",
    ):
        if hg_root is None:
            print(f"Setting default hg_root to {DEFAULT_HG_ROOT}")
            hg_root = DEFAULT_HG_ROOT
        self.hg_root = hg_root
        self.cache_dir = cache_dir
        self.device = device

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
            dtype=torch.float16
        )

    def load_pipe(
        self,
        **kwargs,
    ):
        raise NotImplementedError(
            "This is an abstract method. Implement in a subclass."
        )

    def generate(
        self,
        cloth_image: os.PathLike | Image.Image,
        cloth_mask_image: os.PathLike | Image.Image,
        **kwargs,
    ):
        raise NotImplementedError(
            "This is an abstract method. Implement in a subclass."
        )
