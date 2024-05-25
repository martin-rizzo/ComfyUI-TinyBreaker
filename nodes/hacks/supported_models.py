"""
  File     : supported_models.py
  Brief    : A PixArt-adapted version of 'comfy/supported_models.py' code,
             it offers similar functionality but with a different structure.
  Author   : Martin Rizzo | <martinrizzo@gmail.com>
  Date     : May 10, 2024
  Repo     : https://github.com/martin-rizzo/ComfyUI-PixArt
  License  : MIT
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
from comfy import model_base
from comfy import supported_models_base
from comfy import latent_formats
from comfy import conds
from ...core.pixartsigma import PixArtSigma


#============================================================================
# Clase base similar a la definida en '/comfy/supported_models_base.py'
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models_base.py

class BASE(supported_models_base.BASE):
    unet_extra_config = { }

    def __init__(self, image_size):
        super().__init__( self.__class__.unet_config )
        self.unet_config['input_size']       = image_size//8
        self.unet_config['pe_interpolation'] = image_size//512

    def model_type(self, state_dict, prefix=''):
        return model_base.ModelType.EPS

    def process_unet_state_dict(self, state_dict):
        state_dict, missing_keys = PixArtSigma.get_pixart_state_dict(state_dict)
        if len(missing_keys) > 0:
            print(f"## PixArt DiT conversion has {len(missing_keys)} missing keys!")
            for i, key in enumerate(missing_keys):
                if i>4: print("##    ....") ; break
                print("##    -", key)
            print()
        return state_dict

    def get_model(self, state_dict, prefix='', device=None):
        return PixArt_Model(
                model_config = self,
                model_type   = self.model_type(state_dict, prefix),
                device       = device
                )


#============================================================================
# Clases de configuracion similares a las definidas in '/comfy/supported_models.py'
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py

class PixArtSigma_ModelConfig(BASE):
    target                     = 'PixArtSigma'
    latent_format              = latent_formats.SDXL
    supported_inference_dtypes = [torch.float16, torch.float32]
    unet_config   = {
            'disable_unet_model_creation': True,
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
    sampling_settings = {
            'beta_schedule' : 'sqrt_linear',
            'linear_start'  : 0.0001,
            'linear_end'    : 0.02,
            'timesteps'     : 1000,
            }

    def __init__(self, image_size):
        super().__init__(image_size)


class PixArtAlpha_ModelConfig(BASE):
    target                     = '???'
    latent_format              = latent_formats.SD15
    supported_inference_dtypes = [torch.float16, torch.float32]
    unet_config   = {
            'disable_unet_model_creation': True,
            'input_size'      :    -1, #  <---- sera configurado en __init__
            'pe_interpolation':    -1, #  <---- sera configurado en __init__
            'micro_condition' : False, #  <---- sera configurado en __init__
            'context_len'     :   120, #  120 tokens maximo en el prompt
            'input_dim'       :     4, #    4 channels in latent image
            'hidden_dim'      :  1152, # 1152 channels usados internamente
            'context_dim'     :  4096, # 4096 features por cada prompt token
            'patch_size'      :     2,
            'num_heads'       :    16,
            'depth'           :    28,
            }
    sampling_settings = {
            'beta_schedule' : 'sqrt_linear',
            'linear_start'  : 0.0001,
            'linear_end'    : 0.02,
            'timesteps'     : 1000,
            }

    def __init__(self, image_size):
        super().__init__(image_size)
        self.unet_config['micro_condition'] = (image_size==1024)


#===========================================================================#
# A class similar to the classes defined in '/comfy/model_base.py'
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py

class PixArt_Model(model_base.BaseModel):

    def __init__(self,
                 model_config: BASE,
                 model_type  : model_base.ModelType,
                 device      : torch.device
                 ):
        super().__init__(model_config, model_type, device=device)


        ## DEBUG
        print("## unet_config (DiT)")
        for prop, value in model_config.unet_config.items():
            print(f"##    - {prop:<20}: {value}")
        print("## sampling_settings")
        for prop, value in model_config.sampling_settings.items():
            print(f"##    - {prop:<20}: {value}")
        print()

        unet_config = model_config.unet_config

        if model_config.target == 'PixArtSigma':
            self.diffusion_model = PixArtSigma(**unet_config,
                                               device=device,
                                               frozen=True
                                               )
            self.diffusion_model.to( dtype=model_config.unet_config['dtype'] )
        else:
            assert False, f"Unkown target model: '{self.model_config.target}'"


    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out['return_eps_only'] = conds.CONDConstant(True)

        cond_attn_mask = kwargs.get('cond_attn_mask', None)
        if cond_attn_mask is not None:
            out['context_mask'] = conds.CONDRegular(cond_attn_mask)

        return out
