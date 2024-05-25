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
from safetensors       import safe_open as open_safetensors
from safetensors.torch import save_file as save_safetensors
from .utils            import PROMPT_EMBEDS_DIR

#===========================================================================#
class SavePromptEmbedding:

    #-- initialization --#
    CATEGORY    = 'PixArt/testing'
    TITLE       = '[PixArt] Save Prompt Embedding'
    OUTPUT_NODE = True
    def __init__(self):
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
    FUNCTION     = 'save'
    RETURN_TYPES = ()
    def save(self, positive, negative, filename):

        # positive = [ [cond,{extra_conds}]], ... ]
        # negative = [ [cond,{extra_conds}]], ... ]

        tensors = { }
        for i, embedding_info in enumerate(positive):
            tensors[f'positive.{i}'] = embedding_info[0]
            extra_conds              = embedding_info[1]
            if 'cond_attn_mask' in extra_conds:
                tensors[f'positive.attn_mask.{i}'] = extra_conds['cond_attn_mask']

        for i, embedding_info in enumerate(negative):
            tensors[f'negative.{i}'] = embedding_info[0]
            extra_conds              = embedding_info[1]
            if 'cond_attn_mask' in extra_conds:
                tensors[f'negative.attn_mask.{i}'] = extra_conds['cond_attn_mask']

        # armar el path completo
        _, extension = os.path.splitext(filename)
        if not extension:
            filename += self.default_ext
        filepath = PROMPT_EMBEDS_DIR.get_full_path(filename, for_save=True)

        # almacenar como safetensor
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_safetensors(tensors, filepath)
        return { }


#===========================================================================#
class LoadPromptEmbedding:

    #-- initialization --#
    CATEGORY    = 'PixArt/testing'
    TITLE       = '[PixArt] Load Prompt Embedding'

    #-- parameters --#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'embed_name': (PROMPT_EMBEDS_DIR.get_filename_list(), ),
                }
            }

    #-- function --#
    FUNCTION     = 'load'
    RETURN_TYPES = ('CONDITIONING','CONDITIONING')
    RETURN_NAMES = ('positive'    ,'negative'    )
    def load(self, embed_name):
        positives = []
        negatives = []
        embed_path = PROMPT_EMBEDS_DIR.get_full_path(embed_name)
        with open_safetensors(embed_path, framework='pt', device='cpu') as f:
            _keys = f.keys()

            for i in range(1000):
                extra_conds  = {}
                positive_key = f'positive.{i}'
                pos_mask_key = f'positive.attn_mask.{i}'
                if positive_key not in _keys:
                    break
                if pos_mask_key in _keys:
                    extra_conds['cond_attn_mask'] = f.get_tensor(pos_mask_key)
                positives.append( [f.get_tensor(positive_key), extra_conds] )

            for i in range(1000):
                extra_conds  = {}
                negative_key = f'negative.{i}'
                neg_mask_key = f'negative.attn_mask.{i}'
                if negative_key not in _keys:
                    break
                if neg_mask_key in _keys:
                    extra_conds['cond_attn_mask'] = f.get_tensor(neg_mask_key)
                negatives.append( [f.get_tensor(negative_key), extra_conds] )

        return ( positives, negatives )


#===========================================================================#
NODE_CLASS_MAPPINGS = {
    'SavePromptEmbedding (PixArt)' : SavePromptEmbedding,
    'LoadPromptEmbedding (PixArt)' : LoadPromptEmbedding,
}

