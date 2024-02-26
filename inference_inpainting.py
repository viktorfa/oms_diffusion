import os.path
from oms_diffusion.inference_inpainting import InpaintingModel

from PIL import Image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oms diffusion")
    parser.add_argument("--cloth_path", type=str, required=True)
    parser.add_argument("--person_path", type=str, required=True)
    parser.add_argument("--person_mask_path", type=str, required=True)
    parser.add_argument("--face_mask_path", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--hg_root", type=str, required=False)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument(
        "--pipe_path", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE"
    )
    parser.add_argument("--output_path", type=str, default="./output_img")

    args = parser.parse_args()

    device = "cuda"
    inference_model = InpaintingModel(
        hg_root=args.hg_root,
        cache_dir=args.cache_dir,
        device=device,
    )
    inference_model.load_pipe(
        model_path=args.model_path,
        pipe_path=args.pipe_path,
    )

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cloth_image = Image.open(args.cloth_path).convert("RGB")
    person_image = Image.open(args.person_path).convert("RGB")
    person_mask_image = Image.open(args.person_mask_path).convert("L")
    if args.face_mask_path:
        face_mask_image = Image.open(args.face_mask_path).convert("L")

    images, cloth_mask_image, superposed_images = inference_model.generate(
        cloth_image=cloth_image,
        person_image=person_image,
        person_mask_image=person_mask_image,
        face_mask_image=face_mask_image,
        cloth_mask_image=None,
    )

    for i, image in enumerate(images):
        # Assuming garment_image is the output from your pipeline and already loaded
        # Resize garment to match person image dimensions (if necessary)
        superposed_image = superposed_images[i]

        superposed_image.save(
            os.path.join(output_path, "out_combined_" + str(i) + ".png")
        )
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
