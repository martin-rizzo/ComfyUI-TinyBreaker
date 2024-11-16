"""
File    : __init__.py
Purpose : This file is used to register the nodes of the ComfyUI-xPixArt project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
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
from .utils.system import logger

PROJECT_ID                 ="//xPixart"
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


from .nodes.checkpoint_loader import CheckpointLoader
comfy_import_node(CheckpointLoader)

from .nodes.empty_latent_image import EmptyLatentImage
comfy_import_node(EmptyLatentImage)

from .nodes.placeholder_replacer import PlaceholderReplacer
comfy_import_node(PlaceholderReplacer)

from .nodes.t5_encoder import T5TextEncoder
comfy_import_node(T5TextEncoder)

from .nodes.t5_loader import T5Loader
comfy_import_node(T5Loader)

from .nodes.test.load_prompt_embedding import LoadPromptEmbedding
comfy_import_node(LoadPromptEmbedding)

from .nodes.test.save_prompt_embedding import SavePromptEmbedding
comfy_import_node(SavePromptEmbedding)

logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")
