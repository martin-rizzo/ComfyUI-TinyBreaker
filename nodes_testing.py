"""
  File    : nodes_testing.py
  Brief   : Testing nodes used during the development of PixArt N.C.
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : May 4, 2024
  Repo    : https://github.com/martin-rizzo/ComfyUI-PixArt
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      PixArt Node Collection for ComfyUI
             Nodes providing support for PixArt models in ComfyUI

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
import os
import folder_paths
from safetensors       import safe_open as open_safetensors
from safetensors.torch import save_file as save_safetensors

PROMPT_EMBEDS_DIR='prompt_embeds'

if PROMPT_EMBEDS_DIR not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[PROMPT_EMBEDS_DIR] = (
        [os.path.join(folder_paths.get_output_directory(), PROMPT_EMBEDS_DIR)],
        set(['.safetensors'])
        )

#===========================================================================#
class SavePromptEmbedding_PixArtNC:

    #-- initialization --#
    CATEGORY    = "PixArt/testing"
    TITLE       = "[PixArt] Save Prompt Embedding"
    OUTPUT_NODE = True
    def __init__(self):
        self.directory   = folder_paths.get_folder_paths(PROMPT_EMBEDS_DIR)[0]
        self.default_ext = '.safetensors'

    #-- parameters --#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'positive': ('CONDITIONING', ),
                'negative': ('CONDITIONING', ),
                'filename': ('STRING', {'default': 'prompt'})
                }
            }

    #-- function --#
    FUNCTION     = "save"
    RETURN_TYPES = ()
    def save(self, positive, negative, filename):

        # positive = [ [tensor,{something}]], ... ]

        # print("## positive")
        # print(positive)

        tensors = { }
        for i, embedding_info in enumerate(positive):
            print('## embedding_info =', embedding_info)
            tensors[f'positive.{i}'] = embedding_info[0]
        for i, embedding_info in enumerate(negative):
            tensors[f'negative.{i}'] = embedding_info[0]

        # print("## tensors:")
        # print(tensors)

        # armar el path completo
        _, extension = os.path.splitext(filename)
        if not extension:
            filename += self.default_ext
        filepath = os.path.join(self.directory, filename)

        # almacenar como safetensor
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_safetensors(tensors, filepath)
        return { }


#===========================================================================#
class LoadPromptEmbedding_PixArtNC:

    #-- initialization --#
    CATEGORY    = "PixArt/testing"
    TITLE       = "[PixArt] Load Prompt Embedding"

    #-- parameters --#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'embed_name': (folder_paths.get_filename_list(PROMPT_EMBEDS_DIR), ),
                }
            }

    #-- function --#
    FUNCTION     = 'load'
    RETURN_TYPES = ('CONDITIONING','CONDITIONING')
    RETURN_NAMES = ('positive'    ,'negative'    )
    def load(self, embed_name):
        positives = []
        negatives = []
        embed_path = folder_paths.get_full_path(PROMPT_EMBEDS_DIR, embed_name)
        with open_safetensors(embed_path, framework="pt", device="cpu") as f:
            _keys = f.keys()

            if 'positive.0' in _keys:
                positives.append( [f.get_tensor('positive.0'), {}] )
            if 'positive.1' in _keys:
                positives.append( [f.get_tensor('positive.1'), {}] )

            if 'negative.0' in _keys:
                negatives.append( [f.get_tensor('negative.0'), {}] )
            if 'negative.1' in _keys:
                negatives.append( [f.get_tensor('negative.1'), {}] )

        return ( positives, negatives )


#===========================================================================#
NODE_CLASS_MAPPINGS = {
    'SavePromptEmbedding [PixArt N.C.]' : SavePromptEmbedding_PixArtNC,
    'LoadPromptEmbedding [PixArt N.C.]' : LoadPromptEmbedding_PixArtNC,
}

