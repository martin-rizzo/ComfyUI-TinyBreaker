"""
File     : xconfy/objects.py
Purpose  : The ComfyUI objects transmitted across connected nodes.
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
import torch
import comfy.sd
import comfy.utils
import comfy.model_patcher
from   typing         import Union
from   comfy          import model_management
from   .bridge        import create_model_from_safetensors
from   ..core.t5      import T5Tokenizer, T5EncoderModel
from   ..utils.system import logger


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
class VAE(comfy.sd.VAE):
    """
    A class representing a Variational Autoencoder (VAE).

    This class provides a bridge to the `VAE` class definided in comfy.sd module.
    [https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py]
    """

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str         = "",
                        device    : str         = None,
                        config    : dict        = None,
                        dtype     : torch.dtype = None
                        ) -> "VAE":
        """
        Creates an instance of VAE from a given state dictionary.
        Args:
            state_dict   (dict): A dictionary containing the state of the VAE model.
            prefix       (str) : A string indicating a prefix to filter the keys in the state_dict. Defaults to "".
            device       (str) : The device on which the VAE should be loaded. Defaults to None.
            config       (dict): A dictionary containing configuration parameters for the VAE. Defaults to None.
            dtype (torch.dtype): The data type of the tensors in the VAE model. Defaults to None.
        """
        if not prefix:
            return cls(sd=state_dict, device=device, config=config, dtype=dtype)
        
        # ensure that prefix always ends with a dot '.'
        if not prefix.endswith('.'):
            prefix += '.'
        
        # if a prefix is provided, then only the corresponding part needs to be loaded
        sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return cls(sd=sd, device=device, config=config, dtype=dtype)


#===========================================================================#
class Transcoder:

    def __init__(self, model=None, decoder=None, encoder=None):
        self.model   = model
        self.decoder = decoder
        self.encoder = encoder


    def __call__(self, samples):
        """
        Transcode the input samples using the provided model or decoder/encoder.
        """
        return self.transcode(samples)


    def transcode(self, samples):
        """
        Transcode the input samples using the provided model or decoder/encoder.
        """
        if self.model is None and (self.encoder is None or self.decoder is None):
            logger.debug("No transcoder model or encoder/decoder provided, input samples will not be transcoded.")
            return samples

        # using a transcoder model if available (preferable method)
        if self.model is not None:
            return self.model(samples)

        # fallback to using the encoder and decoder
        images = self.decoder.decode(samples)
        print("##>> images.shape:", images.shape)

        # combine batches if there are 5 dimensions in the images tensor
        #   before: (num_batches, batch_size, height, width, channels)
        #   after : (total_batch_size, height, width, channels)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        # remove alpha channels if there are more than 3 channels per pixel
        if images.shape[-1] > 3:
            images = images[:,:,:,:3]

        samples = self.encoder.encode(images)
        print("##>> samples.shape:", samples.shape)
        return samples



    @classmethod
    def from_decoder_encoder(cls,
                             decoder: VAE,
                             encoder: VAE
                            ) -> "Transcoder":
        """
        Creates an instance of Transcoder from a given decoder and encoder.
        Args:
            decoder (VAE): The VAE model used for decoding.
            encoder (VAE): The VAE model used for encoding.
        """
        return cls(decoder=decoder, encoder=encoder)


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


