<div align="center">

# ComfyUI-TinyBreaker
[<img src="https://img.shields.io/badge/CivitAI%3A-TinyBreaker-EEE?labelColor=1971C2&logo=c%2B%2B&logoColor=white" height="22">](https://civitai.com/models/1213728) |
[<img src="https://img.shields.io/badge/Hugging%20Face%3A-TinyBreaker-EEE?labelColor=FFD21E&logo=huggingface&logoColor=000" height="22">](https://huggingface.co/martin-rizzo/TinyBreaker.prototype1)  
[![Platform](https://img.shields.io/badge/platform%3A-ComfyUI-007BFF)](#)
[![License](https://img.shields.io/github/license/martin-rizzo/ComfyUI-TinyBreaker?label=license%3A&color=28A745)](#)
[![Version](https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-TinyBreaker?label=version%3A&color=D07250)](#)
[![Last](https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-TinyBreaker?label=last%20commit%3A)](#)  

<!-- Main Image -->
![TinyBreaker](./docs/img/tinybreaker_grid_v03.jpg)

</div>

**ComfyUI-TinyBreaker** is a collection of custom nodes specifically designed to generate images using the TinyBreaker model. It's actively developed with ongoing improvements. Although still in progress, these nodes are functional and let you experiment with the model and squeeze out all its potential.  

<!--
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
-->

## Required Files

You need to have these two models copied into your ComfyUI application:

- **[tinybreaker_prototype1.safetensors](https://civitai.com/models/1213728) (3.0 GB)**:
    - place the file in the `'ComfyUI/models/checkpoints'` folder.
- **[t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors) (4.9 GB)**:
    - place the file in the `'ComfyUI/models/clip'` folder (or `'ComfyUI/models/text_encoders'`).
    - this model is a versatile text encoder used by FLUX and SD3.5 as well.


_To utilize all features of version v0.3, including the tiny upscaler, you need the **'prototype1'** version of TinyBreaker._


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

<details>
<summary>🛠️ Manual installation instructions. (expand for details)</summary>
.

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
</details>

### Windows Portable Installation

<details>
<summary>🛠️ Windows portable installation instructions. (expand for details)</summary>
.

1. Go to where you unpacked **ComfyUI_windows_portable**,  
   you'll find your `run_nvidia_gpu.bat` file here, confirming the correct location.
3. Press **CTRL + SHIFT + RightClick** in an empty space and select "Open PowerShell window here".
4. Clone the repository into your custom nodes folder using:
   ```
   git clone https://github.com/martin-rizzo/ComfyUI-TinyBreaker .\ComfyUI\custom_nodes\ComfyUI-TinyBreaker
   ```
</details>


## Example Workflow

<table>
  <tr>
    <td width="190px">
      <img src="workflows/ximg/tinybreaker_workflow.png" alt="TinyBreaker Workflow" width="171" height="256">
    </td>
    <td>
      The image contains a reference workflow for using and testing the TinyBreaker model.<br/>
      <i>- to load this workflow, simply drag and drop the image into ComfyUI.</i><br/>
      <i>- other workflows are available in the <b><a href="workflows">workflows directory</a></b>.</i> 
    </td>
  </tr>
</table>


## Features: Unified Prompt

The **"Unified Prompt"** node allows you to edit both your prompt and parameters within a single text area, accelerating the process of refining the generated image. This even eliminates the need to lift your hands from the keyboard.

When using the Unified Prompt node:  
- Begin by typing your desired prompt text as usual.  
- Then, after the prompt, write any necessary parameters, each preceded by a double hyphen (`--`).  
- Use the **CTRL+RIGHT** key to autocomplete the parameter name if you don’t remember it exactly.  
- Use the **CTRL+UP** and **CTRL+DOWN** keys to modify the value of any parameter.  
- Use the standard **CTRL+ENTER** key to launch the image generation.

#### Parameters Supported by the Unified Prompt

| Minor image adjustments                   | Description                                                                 |
|:------------------------------------------|-----------------------------------------------------------------------------|
| **`--no <text>`**                         | Specifies elements that should *not* appear in the image. (negative prompt) |
| **`--refine <text>`**                     | Specifies elements that should be refined.                                  |
| **`--cfg-shift <number>`**                | Modifies the value of the Classifier-Free Guidance (CFG) scale.             |
| **`--image-shift <number>`**              | Modifies minor image details without altering the overall composition.      |
| **`--upscale [on\|off]`**                 | Enables the application of the upscaling process to the final image.        |


| Major image changes                       | Description                                                                 |
|:------------------------------------------|-----------------------------------------------------------------------------|
| **`--seed <number> \| random`**           | Defines a number for initializing the noise generator.                      |
| **`--aspect <ratio>`**                    | Specifies the aspect ratio of the image (e.g., 16:9, 4:3).                  |
| **`--landscape`**                         | Forces landscape orientation, (ratio 3:2 by default).                       |
| **`--portrait`**                          | Forces portrait orientation, (ratio 2:3 by default).                        |
| **`--medium`**                            | Generates medium-sized images instead of the default large size.            |


| Advanced parameters                       | Description                                                                 |
|:------------------------------------------|-----------------------------------------------------------------------------|
| **`--batch-size <number>`**               | Specifies the number of images to generate in a single batch.               |
| **`--detail-level <number> \| <level>`**  | Controls the level of detail applied during image refinement.               |
| **`--upscale-noise <number> \| <level>`** | Adjusts the extra noise level injected during upscaling.                    |

**\<level\>** = none, minimal, low, normal, high, veryhigh, maximum

#### Examples

`--no trees, clouds` `--refine cats ears` `--cfg-shift -1` `--image_shift +2`  
`--seed random` `--aspect 16:9` `--portrait` `--medium`  
`--detail-level normal` `--batch-size 4`

_For more details on these parameters, see ... \[ ! documentation in preparation ! \]_


## Features: Special CTRL Keys  

The **"Unified Prompt"** node includes specialized control keys to streamline parameter editing:

- **CTRL+RIGHT** _[autocomplete]_  
  Begin typing a parameter name with '--' followed by its initial characters (e.g. '`--de`'). Pressing CTRL+RIGHT automatically completes the full parameter name (e.g. '`--detail-level`').
- **CTRL+LEFT/RIGHT** _[parameter cycling]_  
  If you type '`--`' alone, pressing CTRL+LEFT or CTRL+RIGHT cycles through available parameters, moving backward or forward respectively.
- **CTRL+UP/DOWN** _[value adjustment]_  
  Use these keys to increment or decrement the value of a parameter. For example, if your cursor is on '`--seed 20`', pressing CTRL+UP changes it to '`--seed 21`'.
- **CTRL+ENTER** _[generate]_  
  This key is not part of the Unified Prompt node but serves as ComfyUI’s native shortcut to initiate image generation.


## Features: Predefined Style

The **"Select Style"** node allows you to choose an image style. This node injects text into the prompt and adjusts sampler parameters to influence the image generation, refinement, and upscaling process.

These styles are currently in development as I continuously refine them through experimentation with parameter combinations. As a result, they may not always function perfectly or produce results that exactly match the descriptions provided here.

| Style Name             | Description                                                        |
|:-----------------------|--------------------------------------------------------------------|
| __PHOTO__              | Fast photorealistic images with beautiful design.                  |
| __ULTRAPHOTO__         | Realistic images with exceptional detail and clarity.              |
| __DARKFAN80__          | Dark fantasy images with an 80s cinematic style.                   |
| __LITTLE_TOY__         | Minimalist images inspired by small toy aesthetics.                |
| __PAINTING__           | Evocative, textured artworks in the style of classic paintings.    |
| __COMIC__              | Comic art style emphasizing bold lines and expressive figures.     |
| __PIXEL_ART__          | Pixel art images with retro, blocky details.                       |
| __COLOR_INK__          | Vibrant, colorful ink-style drawings.                              |
| __REALISTIC_WAIFU_X__  | Realistic images where a woman is the main subject.                |
| __REALISTIC_WAIFU_Y__  | Realistic images where a woman is the main subject (alternative1). |
| __REALISTIC_WAIFU_Z__  | Realistic images where a woman is the main subject (alternative2). |
| __CUSTOM1__,...        | Reserved for user-defined style.                                   |

A reference workflow for editing and testing custom styles is available in the [workflows directory](workflows).


## Features: Tiny Upscaler

The **"Tiny Upscaler"** node enables upscaling and improving the resolution of images, allowing for the correction of minor imperfections during the process. This node performs upscaling operations quickly while minimizing resource consumption.

Tiny Upscaler utilizes bilinear scaling combined with the embedded refiner in TinyBreaker, a method similar to the "Hires Fix" technique. Additionally, the node incorporates cross-tiling to optimize VRAM usage while almost completely eliminating visible grid artifacts typically generated during the tiling process.

Currently, Tiny Upscaler is experimental and is configured with minimal creativity. It subtly modifies the image to produce a result that looks as though it were originally generated at a higher resolution, while simultaneously attempting to make minimal visual alterations.

The upscaler is available since version v0.3.0 of the nodes and due to the way VEAs are stored, it requires "prototype1" version of TinyBreaker model to function.


## Features: CivitAI/A1111 Image Metadata

The **"Save Image"** node embeds standard ComfyUI workflow metadata into the generated image. More importantly, it also embeds prompt and parameter data in a format compatible with CivitAI and A1111 platforms.

This enables:
- CivitAI to read the prompt used to generate the image when the image is uploaded.
- A wide range of applications to access the prompt and parameters used for image generation.
- The possibility to include additional information compatible with A1111, and the opportunity to implement an extension that could regenerate the image directly from there.

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
  
