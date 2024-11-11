<div align="center">

# ComfyUI-x-PixArt

<p>
<img alt="Platform" src="https://img.shields.io/badge/platform-ComfyUI-33F">
<img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/ComfyUI-x-PixArt?color=11D">
<img alt="Version"  src="https://img.shields.io/github/v/tag/martin-rizzo/ComfyUI-x-PixArt?label=version">
<img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/ComfyUI-x-PixArt?color=33F">
</p>

<!-- Image -->
<!-- ![PixArt Experimental Nodes](./demo_images/pixart_nodes.png) -->
</div>

**ComfyUI-x-PixArt** is a collection of ComfyUI nodes providing support for 'PixArt-Sigma' model. Currently, it remains beta with limited functionality, but the goal is to provide a full set of nodes, enabling users to experiment with this awesome technology.

# Get Started

## Installation
> [!IMPORTANT]
> Ensure you have the last version of [ComfyUi](https://github.com/comfyanonymous/ComfyUI) installed.

### Manually iInstallation on Linux

Open a terminal and execute the following commands:
```
cd <your_comfyui_directory>/custom_nodes
git clone https://github.com/martin-rizzo/ComfyUI-x-PixArt
```

If ComfyUI is using a virtual environment, activate it before installing the dependencies:
```
# You might need to replace '.venv' with the path to your virtual environment
source .venv/bin/activate
```

Then, install the required dependencies using pip:
```
python -m pip install -r ComfyUI-x-PixArt/requirements.txt
```

### Manually Installation on Windows

If you are using the standalone ComfyUI release on Windows, open a command prompt (CMD)
in the "ComfyUI_windows_portable" folder (the one containing the "run_nvidia_gpu.bat" file).

From that folder, execute the following commands:
```
git clone https://github.com/martin-rizzo/ComfyUI-x-PixArt ComfyUI\custom_nodes\ComfyUI-x-PixArt
.\python_embedded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-x-PixArt\requirements.txt
```


## Minimum Files Required

* You also need to have the following files:
  * PixArt-Sigma-XL-2-1024-MS.safetensors
  * t5-xxl-fp16.safetensors
  

## License

Copyright (c) 2024 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.
