<div align="center">

# ConfyUI-TinyBreaker

<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-ComfyUI-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/ConfyUI-TinyBreaker?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/ConfyUI-TinyBreaker?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/ConfyUI-TinyBreaker?color=33F">
</p>

<!-- Image -->
<!-- ![TinyBreaker Experimental Nodes](./demo_images/nodes.png) -->
</div>

**ConfyUI-TinyBreaker** is a collection of ComfyUI nodes designed to support the 'TinyBreaker' model. Currently, it remains in beta with limited functionality, but the goal is to provide a full set of nodes, enabling users to explore this awesome technology.

## Installation
> [!IMPORTANT]
> Ensure you have the latest version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI) installed.


### Manual Installation on Linux

Open a terminal and navigate to your ComfyUI directory:
```bash
cd <your_comfyui_directory>/custom_nodes
git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker
```

If ComfyUI is using a virtual environment, activate it before installing the dependencies:
```bash
# You might need to replace '.venv' with the path to your virtual environment
source .venv/bin/activate
```

Then, install the required dependencies using pip:
```bash
python -m pip install -r ConfyUI-TinyBreaker/requirements.txt
```


### Manually Installation on Windows

If you are using the standalone ComfyUI release on Windows, open a command prompt (CMD)
in the "ComfyUI_windows_portable" folder (the one containing the "run_nvidia_gpu.bat" file).

From that folder, execute the following commands:
```bash
git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker ComfyUI\custom_nodes\ConfyUI-TinyBreaker
.\python_embedded\python.exe -m pip install -r ComfyUI\custom_nodes\ConfyUI-TinyBreaker\requirements.txt
```


## Minimum Files Required

* You also need to have the following files:
  * TinyBreaker_prototype0.safetensors
  * t5_xxl_encoder-FP8.safetensors


## Acknowledgments

I would like to express my sincere gratitude to the developers of PixArt-Σ for their outstanding model. Their contributions have been instrumental in shaping this project and pushing the boundaries of high-quality image generation with minimal resources.

  * [PixArt-Σ GitHub Repository](https://github.com/PixArt-alpha/PixArt-sigma)
  * [PixArt-Σ Hugging Face Model](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
  * [PixArt-Σ arXiv Report](https://arxiv.org/abs/2403.04692)

Additional thanks to Ollin Boer Bohan for the Tiny AutoEncoder models. These models have proven invaluable for their efficient latent image encoding, decoding and transcoding capabilities.

  * [Tiny AutoEncoder GitHub Repository](https://github.com/madebyollin/taesd)
  

## License

Copyright (c) 2024 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.
  
