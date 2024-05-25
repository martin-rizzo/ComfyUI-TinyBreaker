"""
  File     : packs.py
  Brief    : Implements the objects transmitted through threads across connected nodes.
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

 File Summary
 ============
  - Model_pack: The object transmitted through 'MODEL -> model' threads
  - VAE_pack  : The object transmitted through 'VAE -> vae' threads
  - T5_pack   : The object transmitted through 'T5 -> t5' threads
  - Meta_pack : The object transmitted through 'META -> meta' threads

"""
import os
import comfy.utils
import comfy.model_patcher
from typing                 import Union
from safetensors            import safe_open
from comfy                  import model_management
from .utils                 import load_safetensors_header, estimate_model_params
from .hacks.model_detection import model_config_from_dit
from ..core.t5              import T5Tokenizer, T5EncoderModel


#===========================================================================#
class Model_pack(comfy.model_patcher.ModelPatcher):
    # el objeto que es transmitido por los hilos 'MODEL -> model'
    # debe ser compatible con la siguiente estructura:
    #  - class ModelPatcher
    #      - class BaseModel
    #                .diffusion_model = PixArtMS(..)
    #
    #  ModelPatcher: [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py]

    def __init__(self,
                 model,
                 load_device,
                 offload_device,
                 size=0,
                 current_device=None,
                 weight_inplace_update=False
                 ):
        super().__init__(model,
                         load_device=load_device,
                         offload_device=offload_device,
                         size=size,
                         current_device=current_device,
                         weight_inplace_update=weight_inplace_update
                         )

    # pequeña emulacion al comportamiento de 'load_unet_state_dict(sd)'
    # - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py
    @classmethod
    def from_safetensors(cls,
                         safetensors_path,
                         prefix='',
                         weight_inplace_update=False,
                         dtype=None
                         ):

        header       = load_safetensors_header(safetensors_path)
        model_config = model_config_from_dit(header, prefix)

        # obtener informacion relacionada al modelo a cargar
        parameters          = estimate_model_params(safetensors_path, prefix)
        dit_dtype           = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
        initial_load_device = model_management.unet_inital_load_device(parameters, dit_dtype)
        load_device         = model_management.get_torch_device()
        offload_device      = model_management.unet_offload_device()
        manual_cast_dtype   = model_management.unet_manual_cast(dit_dtype, load_device, model_config.supported_inference_dtypes)

        # preconfigurar los dtype del modelo
        model_config.set_inference_dtype(dit_dtype, manual_cast_dtype)

        ## DEBUG
        print()
        print('##', os.path.basename(safetensors_path).split('.')[0])
        print('##    - parameters          :', parameters         )
        print('##    - dit_dtype           :', dit_dtype          )
        print('##    - initial_load_device :', initial_load_device)
        print('##    - load_device         :', load_device        )
        print('##    - offload_device      :', offload_device     )
        print()

        # obtener los parametros desde el archivo safetensors
        state_dict = {}
        safe_device = initial_load_device if isinstance(initial_load_device,str) else initial_load_device.type
        with safe_open(safetensors_path, framework='pt', device=safe_device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # crear el model
        model = model_config.get_model(state_dict, prefix='', device=initial_load_device)

        # cargar los parametros dentro del model
        model.load_model_weights(state_dict, prefix)
        model.diffusion_model.to(dit_dtype)
        model.diffusion_model.freeze()

        # envolver al modelo en el nuevo objeto (derivado de ModelPatcher)
        return Model_pack(
            model,
            size = 0,
            load_device           = load_device,
            offload_device        = offload_device,
            current_device        = initial_load_device,
            weight_inplace_update = weight_inplace_update
            )


#===========================================================================#
class VAE_pack:
    # el objeto que es transmitido por los hilos 'VAE -> vae'
    # debe ser compatible con:
    #  - class VAE
    #    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    #
    def __init__(self):
        self.placeholder = 'VAEPacket'




#===========================================================================#
class T5_pack:
    # el objeto que es transmitido por los hilos 'T5 -> t5'
    # debe ser compatible con:
    #  - class CLIP
    #    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    #

    @classmethod
    def load( cls, model_path, device="cpu", dtype=None ):

        # carga el archivo mediante carga del directorio
        # TODO: mejorar esto!!
        t5_pack = T5_pack(model_dir = os.path.dirname(model_path),
                          device    = device,
                          dtype     = dtype
                          )
        return t5_pack

        # # carga de archivo torch (.pt?)
        # t5_pack = T5Packet( device = device,
        #                     dtype  = dtype
        #                   )
        # return t5_pack

    def __init__(self, filepath: str = None, embedding_dir=None, device='cpu', dtype=None, init=True ):
        if not init:
            return

        max_length = 300 # alpha=120 / sigma=300 !!!

        # size = 0
        # self.init_device    = 'cpu'
        # self.load_device    = model_management.text_encoder_device()
        # self.offload_device = model_management.text_encoder_offload_device()

        size = 0
        self.init_device    = device
        self.load_device    = device
        self.offload_device = 'cpu'

        if '-of-00002.safetensors' in filepath:
            filepath = \
                ['/home/aiman/Models/t5/model-00001-of-00002.safetensors',
                 '/home/aiman/Models/t5/model-00002-of-00002.safetensors']

        self.tokenizer = T5Tokenizer.from_pretrained(
                            max_length    = max_length,
                            embedding_dir = embedding_dir,
                            legacy        = True
                            )
        self.encoder = T5EncoderModel.from_safetensors(
                            filepath,
                            max_length  = max_length,
                            model_class = 'xxl',
                            frozen      = True,
                            device      = self.init_device
                            )
        self.patcher = comfy.model_patcher.ModelPatcher(
                            self.encoder,
                            current_device = self.init_device,
                            load_device    = self.load_device,
                            offload_device = self.offload_device,
                            size           = size,
                            weight_inplace_update = False,
                            )

    # Returns a copy of this pack
    def clone(self):
        new_pack = T5_pack( init=False )
        new_pack.tokenizer = self.tokenizer
        new_pack.encoder   = self.cond_stage_model
        new_pack.patcher   = self.patcher.clone()
        return new_pack

    def tokenize_with_weights(self,
                              text            : Union[str, list],
                              padding         : bool = False,
                              padding_max_size: int  = 0,
                              include_word_ids: bool = False
                              ):
        return self.tokenizer.tokenize_with_weights(text,
                                                    padding=padding,
                                                    padding_max_size=padding_max_size,
                                                    include_word_ids=include_word_ids)

    def encode_with_weights(self, tokens, return_attn_mask=False, return_pooled=False):
        assert return_pooled == False, "'return_pooled = True' isn´t supported"
        self.load_model()
        return self.encoder.encode_with_weights( tokens, return_attn_mask=return_attn_mask )


    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)


    # Copies parameters and buffers from state_dict to this pack
    def load_sd(self, sd):
        return self.encoder.load_state_dict(sd)

    # Returns a dictionary containing references to the whole state of this pack
    def get_sd(self):
        return self.encoder.state_dict()

    def load_model(self):
        if self.load_device != 'cpu':
            model_management.load_model_gpu(self.patcher)
        return self.patcher

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def get_key_patches(self):
        return self.patcher.get_key_patches()

#===========================================================================#
class Meta_pack(dict):

    def from_predefined(mode, size):
        return None


