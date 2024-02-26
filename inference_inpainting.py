import os.path
import pdb
import numpy as np

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image
import argparse

from garment_adapter.garment_diffusion import ClothAdapter


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert (
        image.shape[0:1] == image_mask.shape[0:1]
    ), "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oms diffusion")
    parser.add_argument("--cloth_path", type=str, required=True)
    parser.add_argument("--person_path", type=str, required=True)
    parser.add_argument("--person_mask_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--hg_root", type=str, required=False)
    parser.add_argument(
        "--pipe_path", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE"
    )
    parser.add_argument("--output_path", type=str, default="./output_img")

    args = parser.parse_args()

    device = "cuda"
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")
    person_image = Image.open(args.person_path).convert("RGB")
    person_mask_image = Image.open(args.person_mask_path).convert("RGB")

    inpaint_image = make_inpaint_condition(person_image, person_mask_image)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        dtype=torch.float16
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pipe_path, vae=vae, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    full_net = ClothAdapter(pipe, args.model_path, device, hg_root=args.hg_root)
    images = full_net.generate(cloth_image, image=inpaint_image)
    for i, image in enumerate(images[0]):
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
