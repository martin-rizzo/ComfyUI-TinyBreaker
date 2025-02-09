<div align="center">

# ComfyUI-TinyBreaker
<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-ComfyUI-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/ComfyUI-TinyBreaker?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-TinyBreaker?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-TinyBreaker?color=33F"> |
<a href="https://civitai.com/models/1213728/tinybreaker">
   <img alt="CivitAI"      src="https://img.shields.io/badge/page-CivitAI-00F"></a>
<!--
<a href="https://huggingface.co/martin-rizzo/tinybreaker">
   <img alt="Hugging Face" src="https://img.shields.io/badge/models-HuggingFace-yellow"></a>
-->
</p>

![TinyBreaker](./docs/img/tinybreaker_grid.jpg)

</div>

**ComfyUI-TinyBreaker** is a collection of custom nodes specifically designed to generate images using the TinyBreaker model. It's actively developed with ongoing improvements. Although still in progress, these nodes are functional and allow you to explore the potential of the model.

**TinyBreaker model**  
While still in the prototype stage, the TinyBreaker model stands out for its unique features. To learn more about its strengths and discover upcoming improvements, check out ["What is TinyBreaker?"](docs/tinybreaker.md)


## Table of Contents

1. [Required Files](#required-files)
2. [Node Installation](#node-installation)
   - [Installation via ComfyUI Manager](#installation-via-comfyui-manager)
   - [Manual Installation](#manual-installation)
   - [Manual Installation (Windows Portable Version)](#manual-installation-windows-portable-version)
3. [Prompt Parameters](#prompt-parameters)
   - [Minor Adjustments](#minor-adjustments)
   - [Major Changes](#major-changes)
   - [Examples](#examples)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)


## Required Files

You need to have these two models copied into your ComfyUI application:

- **[tinybreaker_prototype0.safetensors](https://civitai.com/models/1213728) (3.0 GB)**:
    - place the file in the `'ComfyUI/models/checkpoints'` folder.
- **[t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors) (4.9 GB)**:
    - place the file in the `'ComfyUI/models/clip'` folder (or `'ComfyUI/models/text_encoders'`).
    - this model is a versatile text encoder used by FLUX and SD3.5 as well.


## Node Installation
_Ensure you have the latest version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI)._

### Installation via ComfyUI Manager

1. Access the Manager within ComfyUI.
2. Click "Install via GIT URL" and write:
   ```
   https://github.com/martin-rizzo/ComfyUI-TinyBreaker
   ```
3. After installation, restart the ComfyUI application.

### Manual Installation

To manually install the nodes, follow these steps:

1. Open your preferred file explorer or terminal application.
2. Navigate to your ComfyUI directory:
   ```bash
   cd <your_comfyui_directory>
   ```
3. Move into the **custom_nodes** folder and clone the repository:
   ```bash
   cd custom_nodes
   git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker
   ```

### Manual Installation (Windows Portable Version)

For those using the standalone ComfyUI release on Windows:

1. Go to where you unpacked **ComfyUI_windows_portable**. You'll find your `run_nvidia_gpu.bat` file here, confirming the correct location.
2. Press **CTRL+SHIFT+Right click** in an empty space and select "Open PowerShell window here".
3. Clone the repository into your custom nodes folder using:
   ```
   git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker .\ComfyUI\custom_nodes\ComfyUI-TinyBreaker
   ```

## Features

### Unified Prompt

ComfyUI-TinyBreaker introduces a custom node that allows you to input the prompt and parameters all together in a single text area, streamlining the workflow. The prompt is entered as usual, followed by a series of parameters, each prefixed with `--`.

#### Prompt Parameters
_For more details on these parameters, see [docs/prompt_parameters.md](docs/prompt_parameters.md)._

##### Minor Adjustments

| Parameter                                      | Description                                                                |
|------------------------------------------------|----------------------------------------------------------------------------|
| **`--no <text>`**                              | Specifies elements that should not appear in the image. (negative prompt)  |
| **`--refine <text>`**                          | Provides a textual description of what elements should be refined.         |
| **`--variant <number>`**                       | Specifies variants of the refinement without changing composition.         |
| **`--cfg-adjust <decimal>`**                   | Adjusts the value of the Classifier-Free Guidance (CFG).                   |
| **`--detail <level>`**                         | Sets the intensity level for detail refinement.                            |

##### Major Changes

| Parameter                                      | Description                                                                |
|------------------------------------------------|----------------------------------------------------------------------------|
| **`--seed <number>`**                          | Defines a number for initializing the random generator.                    |
| **`--aspect <ratio>`**                         | Specifies the aspect ratio of the image.                                   |
| **`--landscape`** / **`--portrait`**           | Specifies orientation of the image (horizontal or vertical).               |
| **`--small`** / **`--medium`** / **`--large`** | Controls generated image size.                                             |
| **`--batch-size <number>`**                    | Specifies number of images to generate in a batch.                         |
| **`--style <style>`**                          | Defines the artistic style of the image.                                   |

##### Examples

`--no trees, clouds` `--refine cats ears` `--variant 2` `--cfg-adjust -0.2` `--detail normal`  
`--seed 42` `--aspect 16:9` `--portrait` `--medium` `--batch-size 4` `--style PIXEL_ART`


#### Special Keys

The Unified Prompt text area supports special keys to simplify parameter input:

*   **`--` CTRL+RIGHT:**  Auto-completes available parameters one by one. Use CTRL+RIGHT/LEFT to navigate the list of available parameters.
*   **`--<letter>` CTRL+RIGHT:** If you type `--` followed by the beginning of a parameter name (e.g., `--d`), pressing CTRL+RIGHT will auto-complete the full parameter name (e.g., `--detail`).
*   **CTRL+UP/DOWN (over a parameter value):**  Increments or decrements the numerical value associated with the parameter. For example, placing the cursor over `--seed 20` and pressing CTRL+UP will change the text to `--seed 21`.


### Styles

The __'Select Style'__ node allows you to select an image style. This node injects text into the prompt and modifies sampler parameters to influence the image generation. Please note that these styles are still in development, as I am experimenting with different parameter combinations to refine them over time. Therefore, they might not always function perfectly or reflect exactly what is described here.

#### Available Styles
| Style Name           | Description                                                    |
|----------------------|----------------------------------------------------------------|
| `PHOTO`              | Realistic images that closely resemble photographs.            |
| `DARKFAN80`          | Dark fantasy images with 80s cinematic style.                  |
| `LITTLETOY`          | Cute, minimalist images in the style of small toys.            |
| `PIXEL_ART`          | Pixel art images with retro and blocky details.                |
| `COLOR_INK`          | Beautiful drawings in vibrant colorful ink style.              |
| `REALISTIC_WAIFU_X`  | Realistic images where a woman is the main subject.            |
| `REALISTIC_WAIFU_Z`  | Realistic images where a woman is the main subject (variant)   |


### CivitAI/A1111 Image Compatibility

The __'Save Image'__ node embeds workflow information into the generated image. Additionally, it embeds prompt and parameter information in a format compatible with CivitAI/A1111, this enables:

  * CivitAI can read the prompt used to generate the image when uploaded.
  * A wide range of A1111-compatible applications can access the prompt and parameters used for image generation.


## Acknowledgments

I would like to express my sincere gratitude to the developers of PixArt-Σ for their outstanding model. Their contributions have been instrumental in shaping this project and pushing the boundaries of high-quality image generation with minimal resources.

  * [PixArt-Σ GitHub Repository](https://github.com/PixArt-alpha/PixArt-sigma)
  * [PixArt-Σ Hugging Face Model](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)
  * [PixArt-Σ arXiv Report](https://arxiv.org/abs/2403.04692)

Additional thanks to Ollin Boer Bohan for the Tiny AutoEncoder models. These models have proven invaluable for their efficient latent image encoding, decoding and transcoding capabilities.

  * [Tiny AutoEncoder GitHub Repository](https://github.com/madebyollin/taesd)
  

## License

Copyright (c) 2024-2025 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.
  
