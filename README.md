<div align="center">

# ComfyUI-xPixArt

<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-ComfyUI-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/ComfyUI-xPixArt?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-xPixArt?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-xPixArt?color=33F">
</p>

<!-- Image -->
<!-- ![PixArt Experimental Nodes](./demo_images/pixart_nodes.png) -->
</div>

**ComfyUI-xPixArt** is a collection of ComfyUI nodes designed to support the 'PixArt-Sigma' model. Currently, it remains in beta with limited functionality, but the goal is to provide a full set of nodes, enabling users to explore this awesome technology.

## Installation
> [!IMPORTANT]
> Ensure you have the latest version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI) installed.


### Manual Installation on Linux

Open a terminal and navigate to your ComfyUI directory:
```bash
cd <your_comfyui_directory>/custom_nodes
git clone https://github.com/martin-rizzo/ComfyUI-xPixArt
```

If ComfyUI is using a virtual environment, activate it before installing the dependencies:
```bash
# You might need to replace '.venv' with the path to your virtual environment
source .venv/bin/activate
```

Then, install the required dependencies using pip:
```bash
python -m pip install -r ComfyUI-xPixArt/requirements.txt
```


### Manually Installation on Windows

If you are using the standalone ComfyUI release on Windows, open a command prompt (CMD)
in the "ComfyUI_windows_portable" folder (the one containing the "run_nvidia_gpu.bat" file).

From that folder, execute the following commands:
```bash
git clone https://github.com/martin-rizzo/ComfyUI-xPixArt ComfyUI\custom_nodes\ComfyUI-xPixArt
.\python_embedded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-xPixArt\requirements.txt
```


## Minimum Files Required

* You also need to have the following files:
  * PixArt-Sigma-XL-2-1024-MS.safetensors
  * t5-xxl-fp16.safetensors


## Acknowledgments

I would like to express my sincere gratitude to the developers of PixArt-Σ for their outstanding model. Their contributions have been instrumental in shaping this project and pushing the boundaries of high-quality image generation with minimal resources.

  * [PixArt-Σ GitHub Repository](https://github.com/PixArt-alpha/PixArt-sigma)
  * [PixArt-Σ Hugging Face Model](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
  * [PixArt-Σ arXiv Report](https://arxiv.org/abs/2403.04692)

Additional thanks to Ollin Boer Bohan for the Tiny AutoEncoder models. These models have proven invaluable for their efficient latent image encoding and decoding capabilities.

  * [Tiny AutoEncoder GitHub Repository](https://github.com/madebyollin/taesd)
  

## License

Copyright (c) 2024 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.
  
