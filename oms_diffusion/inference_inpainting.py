import os
from oms_diffusion.inference import AbstractInferenceModel
import numpy as np
import torch
from pathlib import Path
from diffusers import UniPCMultistepScheduler, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
from PIL import Image

from oms_diffusion.garment_adapter.garment_diffusion import ClothAdapter
from .utils.utils import make_inpaint_condition


DEFAULT_HG_ROOT = Path(os.getcwd()) / "oms_models"


class InpaintingModel(AbstractInferenceModel):
    def load_pipe(
        self,
        model_path: str = None,
        pipe_path: str = "SG161222/Realistic_Vision_V4.0_noVAE",
        oms_diffusion_checkpoint: str = "oms_diffusion_100000.safetensors",
    ):
        self.control_net_openpose = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pipe_path,
            vae=self.vae,
            torch_dtype=torch.float16,
            controlnet=self.control_net_openpose,
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
        person_image: os.PathLike | Image.Image,
        person_mask_image: os.PathLike | Image.Image,
        face_mask_image: os.PathLike | Image.Image = None,
        cloth_mask_image: os.PathLike | Image.Image = None,
        use_face_mask: bool = True,
        no_superpose: bool = False,
        **kwargs,
    ):
        if not isinstance(cloth_image, Image.Image):
            cloth_image = Image.open(cloth_image).convert("RGB")
        if not isinstance(person_image, Image.Image):
            person_image = Image.open(person_image).convert("RGB")
        if not isinstance(person_mask_image, Image.Image):
            person_mask_image = Image.open(person_mask_image).convert("L")
        if face_mask_image is not None and not isinstance(face_mask_image, Image.Image):
            face_mask_image = Image.open(face_mask_image).convert("L")
        if cloth_mask_image is not None and not isinstance(
            cloth_mask_image, Image.Image
        ):
            cloth_mask_image = Image.open(cloth_mask_image).convert("L")

        inpaint_image = make_inpaint_condition(person_image, person_mask_image)

        images, cloth_mask_image = self.full_net.generate(
            cloth_image=cloth_image,
            cloth_mask_image=cloth_mask_image,
            inpaint_image=inpaint_image,
            **kwargs,
        )

        # The face can get severely distorted or not look like the input person.
        superposed_images = []
        if no_superpose:
            return images, cloth_mask_image, images
        else:
            if not face_mask_image or no_superpose:
                use_face_mask = False
            mask_image = (
                (face_mask_image if use_face_mask else person_mask_image)
                .resize((384, 512), Image.LANCZOS)
                .convert("L")
            )  # Grayscale
            person_image_resized = person_image.resize(
                (384, 512), Image.LANCZOS
            ).convert("RGBA")
            mask_array = np.array(mask_image)
            binary_mask = (
                np.where(mask_array < 128, 255, 0).astype(np.uint8)
                if use_face_mask
                else np.where(mask_array > 128, 255, 0).astype(np.uint8)
            )
            for image in images:
                result_image = image.convert("RGBA")
                result_image.putalpha(Image.fromarray(binary_mask))
                superposed_image = Image.alpha_composite(
                    person_image_resized, result_image
                )
                superposed_images.append(superposed_image)

        return images, cloth_mask_image, superposed_images
