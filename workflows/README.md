# TinyBreaker Example Workflows

## Setup Instructions

To use these workflows, ensure you have the TinyBreaker custom nodes installed along with these two models:

- **[tinybreaker_prototype0.safetensors](https://civitai.com/models/1213728)**: Place this file in the `ComfyUI/models/checkpoints` folder.
- **[t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors)**: This text encoder, used for FLUX and SD3.5 as well, should be installed in the `ComfyUI/models/clip` folder (or alternatively in `ComfyUI/models/text_encoders`).


## TinyBreaker Workflow

> [!NOTE]
> You can drag this image into ComfyUI to get the workflow.

<img src="ximg/tinybreaker_example.png" width="20%">

This is a simple yet powerful workflow that allows you to adjust parameters within the prompt text box.  
After entering your initial prompt, you can add any of the following parameters:

- `--no <text>`: Exclude specific elements (negative prompt)
- `--refine <text>`: Refine certain aspects of the image.
- `--cfg-shift <number>`: Adjust CFG scale (the value es relative, 0.0 means default CFG)
- `--image-shift <number>`: Choose a variant number for minor changes.
- `--upscale <number>`: Set the upscale factor (e.g., 3 for x3)
- `--seed <number>`: Specify a seed number.
- `--aspect <ratio>`: Define aspect ratio of the image (e.g., 16:9)
- `--portrait`: Choose portrait orientation.
- `--landscape`: Choose landscape orientation.
- `--medium`: Generates medium-sized images.
- `--detail-level <none|low|normal|high>`: Set detail level (normal is recommended)
- `--batch-size <number>`: Set batch processing size.


## TinyBreaker Fully Customizable Workflow (Advanced)

> [!NOTE]
> You can drag this image into ComfyUI to get the workflow.

<img src="ximg/tinybreaker_fully_customizable_example.png" width="20%">

This workflow allows you to customize all elements and alter node connections entirely, enabling full experimentation with the model as desired.
