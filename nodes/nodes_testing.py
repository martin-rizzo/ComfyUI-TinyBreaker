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
import torch
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
MAX_RESOLUTION=16384

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
                'in_positive': ('CONDITIONING', ),
                'in_negative': ('CONDITIONING', ),
                'in_width'   : (   'INT', {'default': 1024, 'min': 1  , 'max': MAX_RESOLUTION, 'step': 1}),
                'in_height'  : (   'INT', {'default': 1024, 'min': 1  , 'max': MAX_RESOLUTION, 'step': 1}),
                'in_seed'    : (   'INT', {'default':    0, 'min': 0  , 'max': 0xffffffffffffffff}),
                'in_steps'   : (   'INT', {'default':   16, 'min': 1  , 'max': 10000         , 'step': 1}),
                'in_cfg'     : ( 'FLOAT', {'default':  4.0, 'min': 0.0, 'max': 100.0, 'step': 0.1, 'round': 0.01}),
                'filename': ('STRING', {'default': 'prompt'})
                }
            }

    #-- function --#
    FUNCTION     = 'save'
    RETURN_TYPES = ()
    def save(self, in_positive, in_negative, in_width, in_height, in_seed, in_steps, in_cfg, filename):

        # positive = [ [cond,{extra_conds}]], ... ]
        # negative = [ [cond,{extra_conds}]], ... ]
        tensors = { }

        multiple = len(in_positive)>1
        for i, cond_unit in enumerate(in_positive):
            prompt_key = f'prompt.positive{i}' if multiple else 'prompt.positive'
            tensors[prompt_key] = cond_unit[0] # 0: cond
            extra_conds         = cond_unit[1] # 1: extra_conds
            if 'cond_attn_mask' in extra_conds:
                attn_mask_key = f"{prompt_key}_attn_mask"
                tensors[attn_mask_key] = extra_conds['cond_attn_mask']

        multiple = len(in_negative)>1
        for i, cond_unit in enumerate(in_negative):
            prompt_key = f'prompt.negative{i}' if multiple else 'prompt.negative'
            tensors[prompt_key] = cond_unit[0] # 0: cond
            extra_conds         = cond_unit[1] # 1: extra_conds
            if 'cond_attn_mask' in extra_conds:
                attn_mask_key = f"{prompt_key}_attn_mask"
                tensors[attn_mask_key] = extra_conds['cond_attn_mask']

        tensors['parameters.size']  = torch.tensor([in_height, in_width])
        tensors['parameters.seed']  = torch.tensor([in_seed])
        tensors['parameters.steps'] = torch.tensor([in_steps])
        tensors['parameters.cfg'  ] = torch.tensor([in_cfg])

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
        embed_path = PROMPT_EMBEDS_DIR.get_full_path(embed_name)
        with open_safetensors(embed_path, framework='pt', device='cpu') as f:
            _keys = f.keys()

            # 'prompt.positive' and 'prompt.positive{1..1000}'
            #   - prompt.positive
            #   - prompt.positive_attn_mask
            cond_unit = self.read_cond_unit('prompt.positive', f, _keys)
            if cond_unit:
                positives.append(cond_unit)
            for i in range(1, 1000):
                cond_unit = self.read_cond_unit(f"prompt.positive{i}", f,_keys)
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
                cond_unit = self.read_cond_unit(f"prompt.negative{i}", f,_keys)
                if not cond_unit:
                    break
                negatives.append(cond_unit)

            # 'parameters.'
            height = 1408
            width  =  944
            seed   = int(  self.read_number('parameters.seed' ,   0, f,_keys))
            steps  = int(  self.read_number('parameters.steps',  16, f,_keys))
            cfg    = float(self.read_number('parameters.cfg'  , 4.0, f,_keys))
            if 'parameters.size' in _keys:
                size = f.get_tensor('parameters.size')
                if size.shape == (2,):
                    height = int(size[0])
                    width  = int(size[1])

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

