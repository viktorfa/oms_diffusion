import copy
from threading import local
import torch
from pathlib import Path
import os
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from garment_seg.process import load_seg_model, generate_mask
from utils.utils import is_torch2_available, prepare_image, prepare_mask

if is_torch2_available():
    from .attention_processor import REFAttnProcessor2_0 as REFAttnProcessor
    from .attention_processor import AttnProcessor2_0 as AttnProcessor

else:
    from .attention_processor import REFAttnProcessor, AttnProcessor

DEFAULT_HG_ROOT = Path(os.getcwd()) / "oodt_models"


class ClothAdapter:
    def __init__(
        self, sd_pipe, ref_path, device, set_seg_model=True, hg_root: str = None
    ):
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_adapter(self.pipe.unet, "write")
        if hg_root is None:
            print(f"Setting default hg_root to {DEFAULT_HG_ROOT}")
            hg_root = DEFAULT_HG_ROOT
        self.hg_root = hg_root

        ref_unet = copy.deepcopy(sd_pipe.unet)
        state_dict = {}
        if not ref_path:
            ref_path = f"{self.hg_root}/checkpoints/oms_diffusion_100000.safetensors"
        if not Path(ref_path).exists():
            print("Downloading reference model")
            hf_hub_download(
                repo_id="shinehugging/oms-diffusion",
                filename="oms_diffusion_100000.safetensors",
                local_dir=f"{self.hg_root}/checkpoints",
            )
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)

        self.ref_unet = ref_unet.to(self.device)
        self.set_adapter(self.ref_unet, "read")
        if set_seg_model:
            self.set_seg_model()
        self.attn_store = {}

    def set_seg_model(
        self,
    ):
        checkpoint_path = f"{self.hg_root}/checkpoints/cloth_segm.pth"
        if not Path(checkpoint_path).exists():
            print("Downloading cloth segmentation model")
            hf_hub_download(
                repo_id="shinehugging/oms-diffusion",
                filename="cloth_segm.pth",
                local_dir=f"{self.hg_root}/checkpoints",
            )
        self.seg_net = load_seg_model(checkpoint_path, device=self.device)

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
        self,
        cloth_image,
        cloth_mask_image=None,
        prompt=None,
        a_prompt="best quality, high quality",
        num_images_per_prompt=4,
        negative_prompt=None,
        seed=-1,
        guidance_scale=2.5,
        num_inference_steps=20,
        height=512,
        width=384,
        **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(
                cloth_image, net=self.seg_net, device=self.device
            )

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = (
                "monochrome, lowres, bad anatomy, worst quality, low quality"
            )

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = self.pipe.encode_prompt(
                [""],
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )[0]
            cloth_embeds = (
                self.pipe.vae.encode(cloth).latent_dist.mode()
                * self.pipe.vae.config.scaling_factor
            )
            self.ref_unet(
                torch.cat([cloth_embeds] * num_images_per_prompt),
                0,
                prompt_embeds_null,
                cross_attention_kwargs={"attn_store": self.attn_store},
            )

        generator = (
            torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        )
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            cross_attention_kwargs={
                "attn_store": self.attn_store,
                "do_classifier_free_guidance": guidance_scale > 1.0,
            },
            **kwargs,
        ).images

        return images, cloth_mask_image
