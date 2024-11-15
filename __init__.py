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
PROJECT_ID="//xPixart"

NODE_CLASS_MAPPINGS = {}

from .nodes.checkpoint_loader import CheckpointLoader
NODE_CLASS_MAPPINGS[f"CheckpointLoader {PROJECT_ID}"] = CheckpointLoader

from .nodes.empty_latent_image import EmptyLatentImage
NODE_CLASS_MAPPINGS[f"EmptyLatentImage {PROJECT_ID}"] = EmptyLatentImage

from .nodes.placeholder_replacer import PlaceholderReplacer
NODE_CLASS_MAPPINGS[f"PlaceholderReplacer {PROJECT_ID}"] = PlaceholderReplacer

from .nodes.t5_encoder import T5TextEncoder
NODE_CLASS_MAPPINGS[f"T5TextEncoder {PROJECT_ID}"] = T5TextEncoder

from .nodes.t5_loader import T5Loader
NODE_CLASS_MAPPINGS[f"T5Loader {PROJECT_ID}"] = T5Loader

from .nodes.test.load_prompt_embedding import LoadPromptEmbedding
NODE_CLASS_MAPPINGS[f"LoadPromptEmbedding {PROJECT_ID}"] = LoadPromptEmbedding

from .nodes.test.save_prompt_embedding import SavePromptEmbedding
NODE_CLASS_MAPPINGS[f"SavePromptEmbeddings {PROJECT_ID}"] = SavePromptEmbedding


NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}

logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")
