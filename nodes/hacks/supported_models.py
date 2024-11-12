"""
File     : supported_models.py
Purpose  : A PixArt-adapted version of 'comfy/supported_models.py' code,
           it offers similar functionality but with a different structure.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 10, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
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
import torch
from comfy.model_base import BaseModel, ModelType
from comfy import supported_models_base
from comfy import latent_formats
from comfy import conds
from ...core.pixartsigma import PixArtSigma


#============================================================================
# Clases de configuracion similares a las definidas in '/comfy/supported_models.py'
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py

class PixArt(supported_models_base.BASE):

    unet_config = {
        'input_size'      :    -1, #  <---- sera configurado en __init__
        'pe_interpolation':    -1, #  <---- sera configurado en __init__
        'context_len'     :   300, #  300 tokens maximo en el prompt
        'input_dim'       :     4, #    4 channels in latent image
        'hidden_dim'      :  1152, # 1152 channels usados internamente
        'context_dim'     :  4096, # 4096 features por cada prompt token
        'patch_size'      :     2,
        'num_heads'       :    16,
        'depth'           :    28,
        }

    unet_extra_config = {
        }

    sampling_settings = {
        'beta_schedule' : 'sqrt_linear',
        'linear_start'  : 0.0001,
        'linear_end'    : 0.02,
        'timesteps'     : 1000,
        }

    latent_format = latent_formats.SDXL
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, image_size):
        super().__init__( self.__class__.unet_config )
        self.unet_config['input_size']       = image_size//8
        self.unet_config['pe_interpolation'] = image_size//512

    def get_model(self, state_dict, prefix='', device=None):
        out = model_base__PixArt(self, device=device)
        return out
    
    def process_unet_state_dict(self, state_dict):
        state_dict, missing_keys = PixArtSigma.get_pixart_state_dict(state_dict)
        if len(missing_keys) > 0:
            print(f"## PixArt DiT conversion has {len(missing_keys)} missing keys!")
            for i, key in enumerate(missing_keys):
                if i>4: print("##    ....") ; break
                print("##    -", key)
            print()
        return state_dict


#===========================================================================#
# A class similar to the classes defined in '/comfy/model_base.py'
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py

class model_base__PixArt(BaseModel):

    def __init__(self,
                 model_config,
                 model_type  : ModelType    = ModelType.EPS,
                 device      : torch.device = None
                 ):
        super().__init__(model_config, model_type, device=device, unet_model=PixArtSigma)


    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out['return_eps_only'] = conds.CONDConstant(True)

        cond_attn_mask = kwargs.get('cond_attn_mask', None)
        if cond_attn_mask is not None:
            out['context_mask'] = conds.CONDRegular(cond_attn_mask)

        return out
