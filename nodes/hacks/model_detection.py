"""
File     : model_detection.py
Purpose  : A PixArt-adapted version of 'comfy/model_detection.py' code,
           it offers similar functionality but with a different structure.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 12, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-x-PixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-x-PixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model

    Copyright (c) 2024 Martin Rizzo

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .supported_models import PixArtSigma_ModelConfig


def model_config_from_dit(state_dict, unet_key_prefix, use_sigma2K_if_no_match=False):
    model_config = PixArtSigma_ModelConfig(image_size=1024)
    if model_config is None and use_sigma2K_if_no_match:
        return PixArtSigma_ModelConfig(image_size=2048)
    else:
        return model_config
