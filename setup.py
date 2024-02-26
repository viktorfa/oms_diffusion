from setuptools import setup, find_packages


setup(
    name="oms_diffusion",
    version="0.0.1",
    description="A description of your project",
    author="Viktor Frede Andersen",
    author_email="vikfand@gmail.com",
    url="https://github.com/viktorfa/oms_diffusion",
    packages=find_packages(include=["oms_diffusion", "oms_diffusion.*"]),
    include_package_data=False,
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "diffusers",
        "opencv-python",
        "transformers",
        "safetensors",
        "controlnet-aux",
        "accelerate",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        "Programming Language :: Python :: 3.10",
    ],
)
