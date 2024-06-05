"""
  File    : nodes_base.py
  Brief   : Main nodes for PixArt N.C.
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
import torch

from .packs import \
    Model_pack,    \
    T5_pack        \
#   VAE_pack,      \
#   Meta_pack      \

from .utils   import        \
    PIXART_CHECKPOINTS_DIR, \
    T5_CHECKPOINTS_DIR


#===========================================================================#
class T5Loader:
    CATEGORY = 'PixArt'
    TITLE    = '[PixArt] T5 Loader'

    #-- parameters --#

    @classmethod
    def INPUT_TYPES(cls):
        devices = ['auto', 'cpu', 'gpu']
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            'required': {
                't5_name' : (T5_CHECKPOINTS_DIR.get_filename_list(), ),
                'device'  : (devices, {'default':'cpu'}),
            }
        }


    #-- function --#

    FUNCTION     = 'load'
    RETURN_TYPES = ('T5',)

    def load(self, t5_name, device):
        # safetensors_path = T5_CHECKPOINTS_DIR.get_full_path(t5_name)
        # t5_pack = T5_pack.from_safetensors(safetensors_path, prefix, device)
        # return (t5_pack,)

        t5_pack = T5_pack(T5_CHECKPOINTS_DIR.get_full_path(t5_name),
                          device = device
                          )
        return (t5_pack,)


#===========================================================================#
class T5TextEncoder:
    CATEGORY = 'PixArt'
    TITLE    = '[PixArt] T5 Text Encoder'

    #-- parameters --#

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text': ('STRING', {'multiline': True}),
                't5'  : ('T5', )
                }
            }

    #-- function --#
    FUNCTION     = 'encode'
    RETURN_TYPES = ('CONDITIONING',)

    def encode(self, t5, text):
        t5_pack = t5
        padding = False

        if padding:
            tokens          = t5_pack.tokenize_with_weights(text,
                                                            padding=True,
                                                            padding_max_size=300)
            cond, attn_mask = t5_pack.encode_with_weights(tokens,
                                                        return_attn_mask=True
                                                        )
        else:
            tokens    = t5_pack.tokenize_with_weights(text, padding=False)
            cond      = t5_pack.encode_with_weights(tokens)
            attn_mask = None


        if attn_mask is not None:
            extra_conds = {'cond_attn_mask':attn_mask}
        else:
            extra_conds = { }

        print("## t5 cond.shape:", cond.shape)
        print("## t5 tokens:", tokens)
        print("## t5 attn_mask:", attn_mask)
        return ([[cond, extra_conds]], )


#===========================================================================#
class CheckpointLoader:
    CATEGORY = 'PixArt'
    TITLE    = '[PixArt] Checkpoint Loader'

    #-- parameters --#

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'ckpt_name': (PIXART_CHECKPOINTS_DIR.get_filename_list(), ),
                }
            }
    RETURN_TYPES = ('MODEL', 'VAE', 'T5', 'STRING')
    RETURN_NAMES = ('MODEL', 'VAE', 'T5', 'META')
    FUNCTION = 'load_checkpoint'

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):

        # safetensors_path = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        # model = Model_packet_.from_safetensors(safetensors_path, prefix)
        # vae   = VAE_packet_.from_safetensors(safetensors_path, prefix)
        # t5    = T5_packet_.from_safetensors(safetensors_path, prefix)
        # meta  = Meta_packet_.from_predefined('sigma', 2048)
        # return (model, vae, t5, meta)


        safetensors_path = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        model_pack = None
        vae_pack   = None
        t5_pack    = None
        meta_pack  = None

        model_pack = Model_pack.from_safetensors(
            safetensors_path,
            prefix = '',
            weight_inplace_update = False
            );

        return (model_pack, vae_pack, t5_pack, meta_pack)


#===========================================================================#
NODE_CLASS_MAPPINGS = {
    'T5Loader (PixArt)'        : T5Loader,
    'T5TextEncoder (PixArt)'   : T5TextEncoder,
    'CheckpointLoader (PixArt)': CheckpointLoader
    }




