<div align="center">

# ComfyUI-TinyBreaker

<!-- Main Image -->
![TinyBreaker](./docs/img/tinybreaker_grid.jpg)

<!-- Badges -->
[![Platform](https://img.shields.io/badge/platform%3A-ComfyUI-007BFF)](#)
[![License](https://img.shields.io/github/license/martin-rizzo/ComfyUI-TinyBreaker?label=license%3A&color=28A745)](#)
[![Version](https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-TinyBreaker?label=version%3A&color=D07250)](#)
[![Last](https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-TinyBreaker?label=last%20commit%3A)](#)  
[![CivitAI](https://img.shields.io/badge/CivitAI%3A-TinyBreaker-EEE?labelColor=1971C2&logo=c%2B%2B&logoColor=white)](https://civitai.com/models/1213728) |
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%3A-TinyBreaker-EEE?labelColor=FFD21E&logo=huggingface&logoColor=000)](https://huggingface.co/martin-rizzo/TinyBreaker.prototype0)

</div>

**ComfyUI-TinyBreaker** is a collection of custom nodes specifically designed to generate images using the TinyBreaker model. It's actively developed with ongoing improvements. Although still in progress, these nodes are functional and allow you to explore the potential of the model.

**TinyBreaker model**  
While still in the prototype stage, the TinyBreaker model stands out for its unique features. To learn more about its strengths and discover upcoming improvements, check out ["What is TinyBreaker?"](docs/tinybreaker.md)


## Table of Contents

1. [Required Files](#required-files)
2. [Node Installation](#node-installation)
   - [Installation via ComfyUI Manager (Recommended)](#installation-via-comfyui-manager-recommended)
   - [Manual Installation](#manual-installation)
3. [Workflow Example](#workflow-example)
4. [Features](#features--unified-prompt)
   - [Unified Prompt](#features--unified-prompt)
   - [Special Ctrl Keys](#features--special-ctrl-keys)
   - [Predefined Styles](#features--predefined-styles)
   - [CivitAI/A1111 Image Compatibility](#features--civitaia1111-image-compatibility)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)


## Required Files

You need to have these two models copied into your ComfyUI application:

- **[tinybreaker_prototype0.safetensors](https://civitai.com/models/1213728) (3.0 GB)**:
    - place the file in the `'ComfyUI/models/checkpoints'` folder.
- **[t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors) (4.9 GB)**:
    - place the file in the `'ComfyUI/models/clip'` folder (or `'ComfyUI/models/text_encoders'`).
    - this model is a versatile text encoder used by FLUX and SD3.5 as well.


## Node Installation
_Ensure you have the latest version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI)._

### Installation via ComfyUI Manager (Recommended)

The easiest way to install the nodes is through ComfyUI Manager:

  1. Open ComfyUI and click on the "Manager" button to launch the "ComfyUI Manager Menu".
  2. Within the ComfyUI Manager, locate and click on the "Custom Nodes Manager" button.
  3. In the search bar, type "tinybreaker".
  4. Select the "ComfyUI-TinyBreaker" node from the search results and click the "Install" button.
  5. Restart ComfyUI to ensure the changes take effect.

### Manual Installation

To manually install the nodes:

1. Open your preferred terminal application.
2. Navigate to your ComfyUI directory:
   ```bash
   cd <your_comfyui_directory>
   ```
3. Move into the **custom_nodes** folder and clone the repository:
   ```bash
   cd custom_nodes
   git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker
   ```

#### Windows Portable

For those using the standalone ComfyUI release on Windows:

1. Go to where you unpacked **ComfyUI_windows_portable**,  
   you'll find your `run_nvidia_gpu.bat` file here, confirming the correct location.
3. Press **CTRL + SHIFT + RightClick** in an empty space and select "Open PowerShell window here".
4. Clone the repository into your custom nodes folder using:
   ```
   git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker .\ComfyUI\custom_nodes\ComfyUI-TinyBreaker
   ```

## Workflow Example

This image contains a simple workflow for testing the TinyBreaker model. To load this workflow, simply drag and drop the image into ComfyUI.

<img src="workflows/ximg/tinybreaker_example.png" width="100px">

_For further information and additional workflow examples, please consult the [workflows folder](workflows)._


## Features : Unified Prompt

The __'Unified Prompt'__ node allows you to input both your prompt and parameters within a single text area, streamlining your workflow. This eliminates the need for separate input fields.

When using the Unified Prompt node:

* Begin by typing your desired prompt text as usual.
* Then write any necessary parameters, each preceded by a double hyphen (`--`).
* Utilize the special keys CTRL+UP and CTRL+DOWN to modify the values of each parameter.


#### Parameters Supported by the Unified Prompt

| Minor image adjustments                | Description                                                                 |
|:---------------------------------------|-----------------------------------------------------------------------------|
| **`--no <text>`**                      | Specifies elements that should *not* appear in the image. (negative prompt) |
| **`--refine <text>`**                  | Specifies elements that should be refined.                                  |
| **`--cfg-shift <number>`**             | Modifies the value of the Classifier-Free Guidance (CFG) scale.             |
| **`--image-shift <number>`**           | Modifies minor image details without altering the overall composition.      |
| **`--upscale`**                        | Enables the application of the upscaling process to the final image.        |


| Major image changes                    | Description                                                                 |
|:---------------------------------------|-----------------------------------------------------------------------------|
| **`--seed <number>`**                  | Defines a number for initializing the random generator.                     |
| **`--aspect <ratio>`**                 | Specifies the aspect ratio of the image (e.g., 16:9, 4:3).                  |
| **`--portrait`**                       | Forces portrait orientation, (ratio 2:3 by default).                        |
| **`--landscape`**                      | Forces landscape orientation, (ratio 3:2 by default).                       |
| **`--medium`**                         | Generates medium-sized images instead of the default large size.            |


| Extra parameters                       | Description                                                                 |
|:---------------------------------------|-----------------------------------------------------------------------------|
| **`--detail-level <level>`**           | Controls the level of detail applied during image refinement.               |
| **`--batch-size <number>`**            | Specifies the number of images to generate in a single batch.               |


#### Examples

`--no trees, clouds` `--refine cats ears` `--cfg-shift -1` `--image_shift 2`  
`--seed 42` `--aspect 16:9` `--portrait` `--medium`  
`--detail-level normal` `--batch-size 4`


_For more details on these parameters, see [docs/prompt_parameters.md](docs/prompt_parameters.md)._


## Features : Special Ctrl Keys

The __'Unified Prompt'__ node offers special control keys for simplifying parameter input and modification:

- **CTRL+RIGHT (autocomplete):**  Initiate a parameter name by typing `--` followed by its beginning (e.g., `--d`). Pressing CTRL+RIGHT will automatically complete the full parameter name (e.g., `--detail-level`).
- **CTRL+UP/DOWN (over parameter value):**  Increment or decrement the value associated with a parameter. For instance, if your cursor is positioned over `--seed 20` and you press CTRL+UP, the text will change to `--seed 21`.


## Features : Predefined Styles

The __'Select Style'__ node allows you to select an image style. This node injects text into the prompt and modifies sampler parameters to influence the image generation. Please note that these styles are still in development, as I am experimenting with different parameter combinations to refine them over time. Therefore, they might not always function perfectly or reflect exactly what is described here.

#### Available Styles
| Style Name           | Description                                                        |
|:---------------------|--------------------------------------------------------------------|
| `PHOTO`              | Fast photorealistic images with beautiful design.                  |
| `ULTRAPHOTO`         | Realistic images with exceptional detail and clarity.              |
| `DARKFAN80`          | Dark fantasy images with 80s cinematic style.                      |
| `LITTLE_TOY`         | Minimalist images in the style of small toys.                      |
| `COMIC_ART`          | Dynamic illustrations in comic book art style.                     |
| `PIXEL_ART`          | Pixel art images with retro and blocky details.                    |
| `COLOR_INK`          | Beautiful drawings in vibrant colorful ink style.                  |
| `REALISTIC_WAIFU_X`  | Realistic images where a woman is the main subject.                |
| `REALISTIC_WAIFU_Y`  | Realistic images where a woman is the main subject (alternative1). |
| `REALISTIC_WAIFU_Z`  | Realistic images where a woman is the main subject (alternative2). |


## Features : CivitAI/A1111 Image Compatibility

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
  
