<div align="center">

# ComfyUI-TinyBreaker

<!-- Badges -->
[![Platform](https://img.shields.io/badge/platform%3A-ComfyUI-007BFF)](#)
[![License](https://img.shields.io/github/license/martin-rizzo/ComfyUI-TinyBreaker?label=license%3A&color=28A745)](#)
[![Version](https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-TinyBreaker?label=version%3A&color=FFC107)](#)
[![Last](https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-TinyBreaker?label=last%20commit%3A)](#)
| [![CivitAI](https://img.shields.io/badge/CivitAI%3A-TinyBreaker-EEE?labelColor=1971C2&logo=c%2B%2B&logoColor=white)](https://civitai.com/models/1213728)
| [![Hugging Face](https://img.shields.io/badge/Hugging%20Face%3A-TinyBreaker-EEE?labelColor=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/martin-rizzo/TinyBreaker.prototype0)

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
3. [Features](#features)
   - [Unified Prompt](#unified-prompt)
   - [Special Ctrl Keys](#special-ctrl-keys)
   - [Styles](#styles)]
   - [CivitAI/A1111 Image Compatibility](#civitaia1111-image-compatibility)
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

The __'Unified Prompt'__ node allows you to input both your prompt and parameters within a single text area, streamlining your workflow. This eliminates the need for separate input fields.

When using the Unified Prompt node:

* Begin by typing your desired prompt text as usual.
* Then write any necessary parameters, each preceded by a double hyphen (`--`).
* Utilize the special keys CTRL+UP and CTRL+DOWN to modify the values of each parameter.


#### Parameters Supported by the Unified Prompt

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

_For more details on these parameters, see [docs/prompt_parameters.md](docs/prompt_parameters.md)._


### Special Ctrl Keys

The __'Unified Prompt'__ node offers special control keys for simplifying parameter input and modification:

- **CTRL+RIGHT (autocomplete):**  Initiate a parameter name by typing `--` followed by its beginning (e.g., `--d`). Pressing CTRL+RIGHT will automatically complete the full parameter name (e.g., `--detail`).
- **CTRL+UP/DOWN (over parameter value):**  Increment or decrement the value associated with a parameter. For instance, if your cursor is positioned over `--seed 20` and you press CTRL+UP, the text will change to `--seed 21`.


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
  * A wide range of applications can access the prompt and parameters used for image generation.


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
  
