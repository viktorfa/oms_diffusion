import os.path
from oms_diffusion.inference_generate import GenerateModel


from PIL import Image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oms diffusion")
    parser.add_argument("--cloth_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--hg_root", type=str, required=False)
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument(
        "--pipe_path", type=str, default="SG161222/Realistic_Vision_V4.0_noVAE"
    )
    parser.add_argument("--output_path", type=str, default="./output_img")

    args = parser.parse_args()

    device = "cuda"
    inference_model = GenerateModel(
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

    images = inference_model.generate(cloth_image)

    for i, image in enumerate(images[0]):
        image.save(os.path.join(output_path, "out_" + str(i) + ".png"))
