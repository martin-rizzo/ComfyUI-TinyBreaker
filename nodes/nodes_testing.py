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
from .utils import PROMPT_EMBEDS_DIR
#
# class safe_open
# ---------------
#  Opens a safetensors lazily and returns tensors as asked
#  Args:
#    filename (str) : The filename to open
#    framework (str): The framework you want you tensors in.
#                     Supported values: `pt`, `tf`, `flax`, `numpy`.
#    device (str)   : The device on which you want the tensors (default 'cpu')
from safetensors       import safe_open as open_safetensors
from safetensors.torch import save_file as save_safetensors


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
    RETURN_TYPES = ('CONDITIONING','CONDITIONING', 'INT'  , 'INT'   , 'INT' ,  'INT' , 'FLOAT')
    RETURN_NAMES = ('positive'    ,'negative'    , 'width', 'height', 'seed', 'steps', 'cfg'  )
    def load(self, embed_name):
        positives = []
        negatives = []
        width     = 944
        height    = 1408
        embed_path = PROMPT_EMBEDS_DIR.get_full_path(embed_name)
        with open_safetensors(embed_path, framework='pt', device='cpu') as f:
            _keys = f.keys()

            # -- new format -------------------------------------

            # 'prompt.positive' and 'prompt.positive{1..1000}'
            #   - prompt.positive
            #   - prompt.positive_attn_mask
            cond_unit = self.read_cond_unit('prompt.positive', f, _keys)
            if cond_unit:
                positives.append(cond_unit)
            for i in range(1, 1000):
                cond_unit = self.read_cond_unit(f"prompt.positive{i}", f, _keys)
                if not cond_unit:
                    break
                positives.append(cond_unit)

            # 'prompt.negative' and 'prompt.negative{1..1000}'
            #   - prompt.negative
            #   - prompt.negative_attn_mask
            cond_unit = self.read_cond_unit('prompt.negative', f, _keys)
            if cond_unit:
                negatives.append(cond_unit)
            for i in range(1, 1000):
                cond_unit = self.read_cond_unit(f"prompt.negative{i}", f, _keys)
                if not cond_unit:
                    break
                negatives.append(cond_unit)

            seed  = int(  self.read_number('parameters.seed' ,   0, f,_keys))
            steps = int(  self.read_number('parameters.steps',  16, f,_keys))
            cfg   = float(self.read_number('parameters.cfg'  , 4.0, f,_keys))

            print("## seed:", seed)
            print("## steps:", steps)
            print("## cfg:", cfg)

            # -- old format --------------------------------------

            # old format
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

        return ( positives, negatives, width, height, seed, steps, cfg )


    def read_cond_unit(self, prompt_key:str, safe_open, safe_keys=None):
        if safe_keys is None:
            safe_keys = safe_open.keys()
        if prompt_key not in safe_keys:
            return None

        attn_mask_key = f"{prompt_key}_attn_mask"

        cond = safe_open.get_tensor(prompt_key)
        cond = cond.unsqueeze(0)
        extra_conds = {}
        if attn_mask_key in safe_keys:
            extra_conds['cond_attn_mask'] = safe_open.get_tensor(attn_mask_key)

        return [cond, extra_conds]

    def read_number(self, number_key, default_number, safe_open, safe_keys=None):
        if safe_keys is None:
            safe_keys = safe_open.keys()
        if number_key not in safe_keys:
            return default_number
        tensor = safe_open.get_tensor(number_key)
        if tensor.numel() != 1:
            return default_number
        return tensor.item()




#===========================================================================#
NODE_CLASS_MAPPINGS = {
    'SavePromptEmbedding (PixArt)' : SavePromptEmbedding,
    'LoadPromptEmbedding (PixArt)' : LoadPromptEmbedding,
}

