"""
File    : xcomfy/helpers/ops.py
Purpose : Custom PyTorch operations for ComfyUI
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 14, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.ops
import torch.nn as nn


class comfy_ops_disable_weight_init(comfy.ops.disable_weight_init):
    """
    This class extends the functionality of `comfy.ops.disable_weight_init`,
    providing seamless pass-through behavior for the missing torch nn
    modules (nn.Sequential, nn.SiLU, etc.).

    The base comfy class was probably designed to facilitate advanced VRAM
    management (--lowvram) in the ComfyUI framework. Think of it as a replacement
    for `torch.nn` within the ComfyUI. It redefines essential PyTorch classes
    like nn.Linear, nn.Conv2d, nn.GroupNorm and others, helping to the ComfyUI's
    system to allocate memory and move tensors from the CPU to GPU as needed.
    """

    class Identity(nn.Identity):
        pass

    class Sequential(nn.Sequential):
        pass

    class ModuleList(nn.ModuleList):
        pass

    class SiLU(nn.SiLU):
        pass

    class ReLU(nn.ReLU):
        pass

    class GELU(nn.GELU):
        pass

    class Upsample(nn.Upsample):
        pass

    class Parameter(nn.Parameter):
        pass


