<div align="center">

# ConfyUI-TinyBreaker

![TinyBreaker Experimental Nodes](./docs/img/banner_nodes.jpg)

<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-ComfyUI-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/ConfyUI-TinyBreaker?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/ConfyUI-TinyBreaker?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/ConfyUI-TinyBreaker?color=33F">
</p>
</div>

**ConfyUI-TinyBreaker** is a collection of custom ComfyUI nodes specifically designed to work with the TinyBreaker model. It is currently under active development, so expect some rough edges and evolving functionality. The nodes are functional, allowing you to explore the potential of the model, but be aware that significant changes are likely between versions as nodes may be completely overhauled.


## What is TinyBreaker?

**TinyBreaker** is a hybrid model that combines the [PixArt model](https://github.com/PixArt-alpha/PixArt-sigma) for base image generation with [Photon model](https://civitai.com/models/84728/photon) (or any SD1 model) for image refinement. The idea is to leverage both models' strengths in these tasks, enabling them to operate efficiently on mid and low-end hardware due to their minimal parameter count. Moreover, by sequentially executing both models, you can offload them to system RAM reducing the VRAM usage. Additionally, TinyBreaker employs [Tiny Autoencoders](https://github.com/madebyollin/taesd) for latent space conversion, optimizing performance and efficiency.


## Installation
> [!IMPORTANT]
> Ensure you have the latest version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI) installed.


### Manual Installation on Linux

Open a terminal and navigate to your ComfyUI directory:
```bash
cd <your_comfyui_directory>
cd custom_nodes
git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker
```

### Manually Installation on Windows

If you are using the standalone ComfyUI release on Windows, open a command prompt (CMD)
in the "ComfyUI_windows_portable" folder (the one containing the "run_nvidia_gpu.bat" file).

From that folder, execute the following commands:
```bash
git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker ComfyUI\custom_nodes\ConfyUI-TinyBreaker
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
  
