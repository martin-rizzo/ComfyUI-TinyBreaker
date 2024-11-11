"""
File    : __init__.py
Purpose : This file is used to register the nodes of the ComfyUI-x-PixArt project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-x-PixArt
License : MIT
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

NODE_CLASS_MAPPINGS = {}

from .nodes.nodes_base import NODE_CLASS_MAPPINGS as main_nodes
NODE_CLASS_MAPPINGS.update(main_nodes)

from .nodes.nodes_extra import NODE_CLASS_MAPPINGS as extra_nodes
NODE_CLASS_MAPPINGS.update(extra_nodes)

from .nodes.nodes_testing import NODE_CLASS_MAPPINGS as testing_nodes
NODE_CLASS_MAPPINGS.update(testing_nodes)

NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"Added {len(NODE_CLASS_MAPPINGS)} nodes from \U0001F3A8\033[1;33mPixArt\033[1;35m-node_collection\033[0m")
