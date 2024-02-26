import os.path
import pdb
import numpy as np

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
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
    control_net_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pipe_path,
        vae=vae,
        torch_dtype=torch.float16,
        controlnet=control_net_openpose,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    full_net = ClothAdapter(pipe, args.model_path, device, hg_root=args.hg_root)
    images = full_net.generate(cloth_image, image=inpaint_image)

    mask_image = (
        Image.open(args.person_mask_path).resize((384, 512), Image.LANCZOS).convert("L")
    )  # Grayscale
    person_image_resized = person_image.resize((384, 512), Image.LANCZOS).convert(
        "RGBA"
    )

    for i, image in enumerate(images[0]):
        # Assuming garment_image is the output from your pipeline and already loaded
        # Resize garment to match person image dimensions (if necessary)
        garment_image = image.convert("RGBA")

        # Convert mask_image to a binary mask (you might need to adjust the threshold)
        mask_array = np.array(mask_image)
        binary_mask = np.where(mask_array > 128, 255, 0).astype(np.uint8)

        # Create an alpha composite image to blend based on the mask
        garment_image.putalpha(Image.fromarray(binary_mask))
        combined_image = Image.alpha_composite(person_image_resized, garment_image)
        combined_image.save(
            os.path.join(output_path, "out_combined_" + str(i) + ".png")
        )
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
