"""
File    : __init__.py
Purpose : Register the nodes of the ComfyUI-TinyBreaker project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
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

PROJECT_ID                 ="//TinyBreaker"
NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def comfy_import_node(cls):
    global NODE_CLASS_MAPPINGS
    global NODE_DISPLAY_NAME_MAPPINGS
    if cls.__name__ in NODE_CLASS_MAPPINGS:
        logger.warning(f"Node class {cls.__name__} already exists, skipping import.")
        return
    comfy_class_name = f"{cls.__name__} {PROJECT_ID}"
    NODE_CLASS_MAPPINGS[comfy_class_name]        = cls
    NODE_DISPLAY_NAME_MAPPINGS[comfy_class_name] = cls.TITLE


# Loader/Builder Nodes

from .nodes.load_checkpoint import LoadCheckpoint
comfy_import_node(LoadCheckpoint)

from .nodes.load_checkpoint_advanced import LoadCheckpointAdvanced
comfy_import_node(LoadCheckpointAdvanced)

from .nodes.load_transcoder import LoadTranscoder
comfy_import_node(LoadTranscoder)

from .nodes.load_any_vae import LoadAnyVAE
comfy_import_node(LoadAnyVAE)

from .nodes.load_partial_vae import LoadPartialVAE
comfy_import_node(LoadPartialVAE)

from .nodes.load_style import LoadStyle
comfy_import_node(LoadStyle)

from .nodes.build_custom_transcoder import BuildCustomTranscoder
comfy_import_node(BuildCustomTranscoder)

# GenParams Nodes

from .nodes.unified_prompt_editor import UnifiedPromptEditor
comfy_import_node(UnifiedPromptEditor)

from .nodes.set_float import SetFloat
comfy_import_node(SetFloat)

from .nodes.set_cfg import SetCFG
comfy_import_node(SetCFG)

from .nodes.set_image import SetImage
comfy_import_node(SetImage)

from .nodes.set_image_cfg_and_seed import SetImageCFGAndSeed
comfy_import_node(SetImageCFGAndSeed)

from .nodes.set_noise_seed import SetNoiseSeed
comfy_import_node(SetNoiseSeed)

# Operator Nodes

from .nodes.empty_latent_image import EmptyLatentImage
comfy_import_node(EmptyLatentImage)

from .nodes.encode_prompts import EncodePrompts
comfy_import_node(EncodePrompts)

from .nodes.gen_params_unpacker import GenParamsUnpacker
comfy_import_node(GenParamsUnpacker)

from .nodes.placeholder_replacer import PlaceholderReplacer
comfy_import_node(PlaceholderReplacer)

from .nodes.prefixed_double_stage_sampler import PrefixedDoubleStageSampler
comfy_import_node(PrefixedDoubleStageSampler)

from .nodes.transcode_latent import TranscodeLatent
comfy_import_node(TranscodeLatent)

from .nodes.transcode_latent_two_steps import TranscodeLatentTwoSteps
comfy_import_node(TranscodeLatentTwoSteps)


# Development Nodes




logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")
