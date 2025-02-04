"""
File    : __init__.py
Purpose : Register the nodes of the ComfyUI-TinyBreaker project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)

    Copyright (c) 2024-2025 Martin Rizzo

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
from .nodes.utils.system import logger
WEB_DIRECTORY              = "./web"
NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_PROJECT_ID    = "//TinyBreaker"
_PROJECT_EMOJI = "💪"
_CATEGORY      = "TinyBreaker"

def comfy_import_node(cls):
    global NODE_CLASS_MAPPINGS
    global NODE_DISPLAY_NAME_MAPPINGS
    if cls.__name__ in NODE_CLASS_MAPPINGS:
        logger.warning(f"Node class {cls.__name__} already exists, skipping import.")
        return
    cls.CATEGORY = f"{_PROJECT_EMOJI}{_CATEGORY}"
    comfy_class_name = f"{cls.__name__} {_PROJECT_ID}"
    NODE_CLASS_MAPPINGS[comfy_class_name]        = cls
    NODE_DISPLAY_NAME_MAPPINGS[comfy_class_name] = cls.TITLE


# TinyBreaker/genparams
_CATEGORY = "TinyBreaker/genparams"

from .nodes.select_style import SelectStyle
comfy_import_node(SelectStyle)

from .nodes.set_float import SetFloat
comfy_import_node(SetFloat)

from .nodes.set_image import SetImage
comfy_import_node(SetImage)

from .nodes.set_image_tweaks import SetImageTweaks
comfy_import_node(SetImageTweaks)

from .nodes.set_seed import SetSeed
comfy_import_node(SetSeed)

from .nodes.unified_prompt_input import UnifiedPromptInput
comfy_import_node(UnifiedPromptInput)


# TinyBreakers/loaders
_CATEGORY = "TinyBreaker/loaders"

from .nodes.load_tinybreaker_checkpoint import LoadTinyBreakerCheckpoint
comfy_import_node(LoadTinyBreakerCheckpoint)

from .nodes.load_tinybreaker_checkpoint_custom import LoadTinyBreakerCheckpointCustom
comfy_import_node(LoadTinyBreakerCheckpointCustom)

from .nodes.load_any_vae import LoadAnyVAE
comfy_import_node(LoadAnyVAE)

from .nodes.load_partial_vae import LoadPartialVAE
comfy_import_node(LoadPartialVAE)


# TinyBreakers/transcoding
_CATEGORY = "TinyBreaker/transcoding"

from .nodes.load_transcoder import LoadTranscoder
comfy_import_node(LoadTranscoder)

from .nodes.build_custom_transcoder import BuildCustomTranscoder
comfy_import_node(BuildCustomTranscoder)

from .nodes.transcode_latent import TranscodeLatent
comfy_import_node(TranscodeLatent)

from .nodes.transcode_latent_two_steps import TranscodeLatentTwoSteps
comfy_import_node(TranscodeLatentTwoSteps)


# TinyBreaker
_CATEGORY = "TinyBreaker"

from .nodes.double_stage_sampler import DoubleStageSampler
comfy_import_node(DoubleStageSampler)

from .nodes.empty_latent_image import EmptyLatentImage
comfy_import_node(EmptyLatentImage)

from .nodes.save_image import SaveImage
comfy_import_node(SaveImage)

from .nodes.unpack_sampler_params import UpackSamplerParams
comfy_import_node(UpackSamplerParams)


# TinyBreaker/__dev
_CATEGORY = "TinyBreaker/__dev"

from .nodes.genparams_debug_logger import GenParamsDebugLogger
comfy_import_node(GenParamsDebugLogger)



logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
