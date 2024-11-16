"""
File     : comfy_objs.py
Purpose  : The objects transmitted across connected nodes in ComfyUI.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 10, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 File Summary
 ============
  - Model_pack: The object transmitted through `MODEL -> model` threads
  - VAE_pack  : The object transmitted through `VAE -> vae` threads
  - T5_pack   : The object transmitted through `T5 -> t5` threads
  - Meta_pack : The object transmitted through `META -> meta` threads

"""
import os
import comfy.utils
import comfy.model_patcher
from   typing          import Union
from   comfy           import model_management
from   .comfy_bridge   import create_model_from_safetensors
from   .core.t5        import T5Tokenizer, T5EncoderModel


#===========================================================================#
class Model_pack(comfy.model_patcher.ModelPatcher):
    # el objeto que es transmitido por los hilos "MODEL -> model"
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
                 weight_inplace_update=False
                 ):
        super().__init__(model,
                         load_device=load_device,
                         offload_device=offload_device,
                         size=size,
                         weight_inplace_update=weight_inplace_update
                         )

    # pequeña emulacion al comportamiento de "load_unet_state_dict(sd)"
    # - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py
    @classmethod
    def from_safetensors(cls,
                         safetensors_path,
                         prefix="",
                         weight_inplace_update=False,
                         dtype=None
                         ):


        load_device    = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        model = create_model_from_safetensors(safetensors_path,
                                              prefix         = prefix,
                                              load_device    = load_device,
                                              offload_device = offload_device)
        model.diffusion_model.freeze()

        # envolver al modelo en el nuevo objeto (derivado de ModelPatcher)
        return Model_pack(
            model,
            size = 0,
            load_device           = load_device,
            offload_device        = offload_device,
            weight_inplace_update = weight_inplace_update
            )


#===========================================================================#
class VAE_pack:
    # el objeto que es transmitido por los hilos "VAE -> vae"
    # debe ser compatible con:
    #  - class VAE
    #    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    #
    def __init__(self):
        self.placeholder = "VAEPacket"




#===========================================================================#
class T5_pack:
    # el objeto que es transmitido por los hilos "T5 -> t5"
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

    def __init__(self, filepath: str = None, embedding_dir=None, device="cpu", dtype=None, init=True ):
        if not init:
            return

        max_length = 300 # alpha=120 / sigma=300 !!!

        # size = 0
        # self.init_device    = "cpu"
        # self.load_device    = model_management.text_encoder_device()
        # self.offload_device = model_management.text_encoder_offload_device()

        size = 0
        self.init_device    = device
        self.load_device    = device
        self.offload_device = "cpu"

        if "-of-00002.safetensors" in filepath:
            filepath = \
                ["/home/aiman/Models/t5/model-00001-of-00002.safetensors",
                 "/home/aiman/Models/t5/model-00002-of-00002.safetensors"]

        self.tokenizer = T5Tokenizer.from_pretrained(
                            max_length    = max_length,
                            embedding_dir = embedding_dir,
                            legacy        = True
                            )
        self.encoder = T5EncoderModel.from_safetensors(
                            filepath,
                            max_length  = max_length,
                            model_class = "xxl",
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
        assert return_pooled == False, "`return_pooled = True` isn´t supported"
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
        if self.load_device != "cpu":
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


