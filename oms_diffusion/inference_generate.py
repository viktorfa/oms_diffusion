import os
from oms_diffusion.inference import AbstractInferenceModel
import torch
from pathlib import Path
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image

from oms_diffusion.garment_adapter.garment_diffusion import ClothAdapter


DEFAULT_HG_ROOT = Path(os.getcwd()) / "oms_models"


class GenerateModel(AbstractInferenceModel):
    def load_pipe(
        self,
        model_path: str = None,
        pipe_path: str = "SG161222/Realistic_Vision_V4.0_noVAE",
        oms_diffusion_checkpoint: str = "oms_diffusion_100000.safetensors",
    ):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pipe_path, vae=self.vae, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.full_net = ClothAdapter(
            self.pipe,
            model_path,
            self.device,
            oms_diffusion_checkpoint=oms_diffusion_checkpoint,
            hg_root=self.hg_root,
            cache_dir=self.cache_dir,
        )

    def generate(
        self,
        cloth_image: os.PathLike | Image.Image,
        cloth_mask_image: os.PathLike | Image.Image,
        **kwargs,
    ):
        if not isinstance(cloth_image, Image.Image):
            cloth_image = Image.open(cloth_image).convert("RGB")
        if cloth_mask_image is not None and not isinstance(
            cloth_mask_image, Image.Image
        ):
            cloth_mask_image = Image.open(cloth_mask_image).convert("L")

        images, cloth_mask_image = self.full_net.generate(
            cloth_image=cloth_image,
            cloth_mask_image=cloth_mask_image,
            **kwargs,
        )

        return images, cloth_mask_image
