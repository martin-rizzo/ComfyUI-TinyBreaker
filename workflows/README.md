# TinyBreaker Example Workflows

## Setup Instructions

To use these workflows, ensure you have the TinyBreaker custom nodes installed along with these two models:

- **[tinybreaker_prototype1.safetensors](https://civitai.com/models/1213728)**: Place this file in the `ComfyUI/models/checkpoints` folder.
- **[t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors)**: This text encoder, used for FLUX and SD3.5 as well, should be installed in the `ComfyUI/models/clip` folder (or alternatively in `ComfyUI/models/text_encoders`).

## Example Workflows

### TinyBreaker Workflow

<table>
  <tr>
    <td width="320px"> <img src="ximg/tinybreaker_workflow.png"  alt="workflow" width="308px" height="463px"></td>
    <td> This is a simple yet powerful workflow that allows you to adjust parameters within the prompt text box.
<br/><br/>
<i>(you can drag the image into ComfyUI to get the workflow)</i>
   </tr> 
</table>


### TinyBreaker Ultimate Control (Advanced)

<table>
  <tr>
    <td width="320px"> <img src="ximg/tinybreaker_ultimate_control.png"  alt="workflow" width="308px" height="463px"></td>
    <td> This workflow enables you to customize every aspect of the image generation process. You can alter or expand entire node connections, making it an ideal foundation for advanced experiments or for repurposing groups of nodes from TinyBreaker in other models and projects.
<br/><br/>
<i>(you can drag the image into ComfyUI to get the workflow)</i>
   </tr> 
</table>

### TinyBreaker Style Creator (Advanced)

<table>
  <tr>
    <td width="320px"> <img src="ximg/tinybreaker_style_creator.png"  alt="workflow" width="308px" height="463px"></td>
    <td> This workflow is designed to experiment with customizing the configuration of each style. It’s a simple tool where every parameter of the diffusion process can be adjusted in a text field within the workflow, allowing you to quickly see the final outcome of that style. As it stands now, it’s not particularly useful for anything other than experimentation. However, it’s the tool I rely on to calibrate each style and to determine which sampler and scheduler perform best.
<br/><br/>
<i>(you can drag the image into ComfyUI to get the workflow)</i>
    </td>
   </tr> 
</table>
