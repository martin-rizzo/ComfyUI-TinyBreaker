"""
File    : xcomfy/clip.py

Purpose : The standard CLIP object transmitted through ComfyUI's node system.
          This CLIP object is directly derived from `comfy.sd.CLIP`, extending
          it to support custom code.

Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 3, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

  Over simplification of the ComfyUI CLIP class for SD3
  -----------------------------------------------------
  clip : CLIP
    > .patcher          : model_patcher.ModelPatcher
    > .tokenizer        : text_encoders.sd3_clip.SD3Tokenizer
    > .cond_stage_model : text_encoders.sd3_clip.SD3ClipModel
                 > .clip_l : ??
                 > .clip_g : ??
                 > .t5xxl  : sd1_clip.SDClipModel
                     > .transformer : nn.Module (the T5 encoder model)

"""
import torch
import comfy.sd
import comfy.utils
import comfy.sd1_clip
import comfy.model_management
import comfy.text_encoders.t5
import comfy.text_encoders.sd3_clip
from   torch         import nn
from   ..system      import logging
from   ..directories import EMBEDDINGS_DIR
from   ..safetensors import filter_state_dict, normalize_prefix
from   .t5.t5_config import T5XXLModel_SD3, T5XXLModel_PA, PixArtTokenizer

_CLIP_TYPES_BY_NAME = {
    "stable_diffusion": comfy.sd.CLIPType.STABLE_DIFFUSION,
    "pixart"          : comfy.sd.CLIPType.PIXART,
    "sd3"             : comfy.sd.CLIPType.SD3,
    # "stable_cascade"  : comfy.sd.CLIPType.STABLE_CASCADE,
    # "stable_audio"    : comfy.sd.CLIPType.STABLE_AUDIO,
    # "mochi"           : comfy.sd.CLIPType.MOCHI,
    # "ltxv"            : comfy.sd.CLIPType.LTXV,
    # "cosmos"          : comfy.sd.CLIPType.COSMOS,
    # "lumina2"         : comfy.sd.CLIPType.LUMINA2,
}


#--------------------------------- HELPERS ---------------------------------#

def _create_pixart_clip_model_class(t5_encoder_class: type[nn.Module],
                                    *,
                                    dtype_t5        : torch.dtype = None,
                                    t5xxl_scaled_fp8: torch.dtype = None,
                                    ) -> type[ comfy.sd1_clip.SD1ClipModel ]:
    """Returns a custom TYPE that must be used to instantiate a CLIP object for PixArt models."""

    class _PixArtT5XXL(comfy.sd1_clip.SD1ClipModel):
        """A custom subclass that attempts to emulate the behavior of PixArtT5XXL/PixArtTEModel_
        but injecting the custom T5 encoder class. The original implementation can be found here:
        https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/text_encoders/pixart_t5.py#L34
        """
        def __init__(self, *,
                     device        = "cpu",
                     dtype         = None,
                     model_options = {},
                     **kwargs
                     ):
            model_options = model_options.copy()
            model_options["t5_encoder_class"] = t5_encoder_class # <-- inject the custom T5 encoder, to be read by T5XXLModel
            if t5xxl_scaled_fp8:
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8

            super().__init__(device        = device,
                             dtype         = dtype_t5,
                             name          = "t5xxl",
                             clip_model    = T5XXLModel_PA,
                             model_options = model_options)

    # return the configured class with the custom T5 encoder injected
    return _PixArtT5XXL


def _create_sd3_clip_model_class(t5_encoder_class : type[torch.nn.Module],
                                 *,
                                 clip_l           : bool        = False,
                                 clip_g           : bool        = False,
                                 t5               : bool        = True,
                                 t5_attention_mask: bool        = False,
                                 dtype_t5         : torch.dtype = None,
                                 t5xxl_scaled_fp8 : torch.dtype = None,
                                 ) -> type[comfy.text_encoders.sd3_clip.SD3ClipModel]:
    """
    Creates a custom subclass of `comfy.text_encoders.sd3_clip.SD3ClipModel`.

    This class enables the creation of a specialized SD3 CLIP model based
    on user-defined parameters, but most notably allowing for the injection
    of a custom T5 encoder.

    Args:
        t5_encoder_class  (type): The class of the T5 encoder model to be injected.
                                  This should be a class inheriting from `torch.nn.Module`.
        clip_l            (bool): Whether to include a CLIP text encoder (L). Defaults to False.
        clip_g            (bool): Whether to include a CLIP text encoder (G). Defaults to False.
        t5                (bool): Whether to use the T5 encoder. Defaults to True.
        t5_attention_mask (bool): Whether to use attention masks in the T5 encoder. Defaults to False.
        dtype_t5   (torch.dtype): The data type for the T5 encoder.
        t5xxl_scaled_fp8 (torch.dtype): The data type for fp8 scaling (None = no scaling).

    Returns:
        A new class that inherits from `comfy.text_encoders.sd3_clip.SD3ClipModel`
        with the specified configurations and custom T5 encoder class. 
    """

    class _SD3ClipModel(comfy.text_encoders.sd3_clip.SD3ClipModel):
        """A custom subclass of `comfy.text_encoders.sd3_clip.SD3ClipModel`
        with the custom T5 encoder class injected. The base class can be found here:
        - https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/text_encoders/sd3_clip.py#L59
        """
        def __init__(self, *,
                     device        = "cpu",
                     dtype         = None,
                     model_options = {},
                     **kwargs
                     ):
            model_options = model_options.copy()
            model_options["t5_encoder_class"] = t5_encoder_class # <-- inject the custom T5 encoder, to be read by T5XXLModel
            if t5xxl_scaled_fp8:
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8

            super().__init__(clip_l            = clip_l,
                             clip_g            = clip_g,
                             t5                = False,
                             dtype_t5          = dtype_t5,
                             t5_attention_mask = t5_attention_mask,
                             device            = device,
                             dtype             = dtype,
                             model_options     = model_options
                             )
            if t5:
                __dtype = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
                self.t5_attention_mask = t5_attention_mask
                self.t5xxl = T5XXLModel_SD3(device         = device,
                                            dtype          = __dtype,
                                            model_options  = model_options,
                                            attention_mask = self.t5_attention_mask
                                            )
                self.dtypes.add(__dtype)

    # return the configured class with the custom T5 encoder injected
    return _SD3ClipModel


#===========================================================================#
#////////////////////////////////// CLIP ///////////////////////////////////#
#===========================================================================#

class CLIP(comfy.sd.CLIP):
    """
    A class representing a Text Encoder.
    (its name originates from the early development stages when only CLIP encoders were available)
    """

    @classmethod
    def from_state_dict(cls,
                        state_dicts: list[dict],
                        *,
                        prefixes   : list[str] = None,
                        prefix     : str       = None,
                        clip_type  : str       = "stable_diffusion",
                        ) -> "CLIP":
        """
        Create an instance of the CLIP class from one or more state dictionaries.

        Args:
            state_dicts: A list of dictionaries containing the state of a CLIP model,
                         or a single dictionary representing a single CLIP model's state.
            prefixes   : A list of prefixes to filter the state dictionaries
                         or a single prefix when a single dictionary is passed.
            prefix     : An alternative way to specify a single prefix for filtering state dictionaries,
                         ensuring backward compatibility with usage patterns.
            clip_type  : The type of CLIP model to create. Defaults to "stable_diffusion".
         """

        # `prefix` is an alternative way of passing prefixes (for backwards compatibility)
        if prefixes is None:
            prefixes =  prefix

        # if only one state_dict/prefix is provided, wrap it in a list
        if isinstance(state_dicts, dict):
            state_dicts = [state_dicts]
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        # normalize all prefixes
        if prefixes:
            prefixes = [ normalize_prefix(prefix) for prefix in prefixes ]


        # resolve `clip_type_object` to a valid CLIPType object
        if isinstance(clip_type, str):
            clip_type_object = _CLIP_TYPES_BY_NAME.get(clip_type)

        if not clip_type_object:
            raise ValueError(f"Unsupported clip_type: {clip_type}")


        # filter state dictionaries based on their prefixes
        # (if only one prefix is provided, it will be applied to all state dictionaries)
        if prefixes:
            if len(prefixes) == 1:
                state_dicts = [ filter_state_dict(sd, prefixes[0]) for sd in state_dicts ]
            elif len(prefixes) == len(state_dicts):
                state_dicts = [ filter_state_dict(sd, prefix) for sd, prefix in zip(state_dicts, prefixes) ]
            else:
                raise ValueError(f"Number of prefixes ({len(prefixes)}) does not match number of state dictionaries ({len(state_dicts)}).")


        # load the CLIP model using the native ComfyUI function
        clip = comfy.sd.load_text_encoder_state_dicts(state_dicts,
                                                      clip_type           = clip_type_object,
                                                      embedding_directory = EMBEDDINGS_DIR.paths)
        return clip


    @classmethod
    def from_components(cls,
                        *,
                        comfy_tokenizer_class  : type,
                        comfy_model_class      : type,
                        number_of_parameters   : int,
                        comfy_tokenizer_options: dict = {},
                        comfy_model_options    : dict = {},
                        model_options          : dict = {},
                        ) -> "CLIP":
        """
        Creates a standard ComfyUI CLIP object from the provided components.
        Args:
            comfy_tokenizer_class: The class of the tokenizer to use.
            comfy_model_class    : The class of the text model to use.
        """
        class EmptyClass:
            pass
        target = EmptyClass()
        target.tokenizer = comfy_tokenizer_class
        target.clip      = comfy_model_class
        target.params    = comfy_model_options
        # call the initializer of the native ComfyUI CLIP class (comfy.sd.CLIP)
        return cls(target,
                   embedding_directory = EMBEDDINGS_DIR.paths,
                   tokenizer_data      = comfy_tokenizer_options,
                   parameters          = number_of_parameters,
                   model_options       = model_options)


    @classmethod
    def from_custom_t5_encoder(cls,
                               t5_encoder_class: type,
                               state_dicts: list[dict], /,
                               *,
                               prefix          : str          = None,
                               clip_type       : str          = "sd3",
                               initial_device  : torch.device = None,
                               load_device     : torch.device = None,
                               offload_device  : torch.device = None,
                               ) -> "CLIP":
        """
        Creates a standard ComfyUI CLIP object from a custom T5 text encoder model.

        This function provides a simplified alternative to the ComfyUI function
        `load_text_encoder_state_dicts(...)`, focused only in the T5 text encoder
        and enabling the use of custom models.

        The original function can be found at:
        - https://github.com/comfyanonymous/ComfyUI/blob/v0.3.18/comfy/sd.py#L740

        Args:
            t5_encoder_class: The class of the custom T5 model (it must be a class, not an instance!)
            state_dicts     : A list of dictionaries containing the model's state.
                              Note that a single dictionary can also be provided!
            prefix          : A tensor name prefix used to filter part of the state dictionary.
            clip_type       : The type of CLIP model to create. Currently supports "sd3" and "pixart".
            initial_device  : The device used to load the model's parameters from storage.
            load_device     : The device used during inference.
            offload_device  : The device used to offload the model while not used.
        Returns:
            A ComfyUI CLIP object initialized with the custom T5 encoder.
        """
        # if not custom T5 model is provided, use the standard ComfyUI T5 raw model
        if not t5_encoder_class:
            t5_encoder_class = comfy.text_encoders.t5.T5

        # if a single state_dict is provided, wrap it in a list
        if isinstance(state_dicts, dict):
            state_dicts = [state_dicts]

        # filter state dictionaries based on the prefix
        prefix = normalize_prefix(prefix)
        if prefix:
            state_dicts = [ filter_state_dict(sd, prefix) for sd in state_dicts ]

        # detect some T5 extra options
        #  - dtype_t5         : the model's data type (float16, bfloat16, float8_e4m3fn, etc)
        #  - t5xxl_scaled_fp8 : data type for fp8 scaling (None = no scaling)
        _extra_params = comfy.sd.t5xxl_detect(state_dicts)
        dtype_t5         = _extra_params.get("dtype_t5")
        t5xxl_scaled_fp8 = _extra_params.get("t5xxl_scaled_fp8")

        # create the ComfyUI model/tokenizer classes
        if clip_type == "sd3":
            comfy_tokenizer_class = comfy.text_encoders.sd3_clip.SD3Tokenizer
            comfy_model_class     = _create_sd3_clip_model_class(t5_encoder_class,
                                                                 clip_l           = False,
                                                                 clip_g           = False,
                                                                 t5               = True,
                                                                 dtype_t5         = dtype_t5,
                                                                 t5xxl_scaled_fp8 = t5xxl_scaled_fp8)

        elif clip_type == "pixart":
            comfy_tokenizer_class = PixArtTokenizer
            comfy_model_class     = _create_pixart_clip_model_class(t5_encoder_class,
                                                                    dtype_t5         = dtype_t5,
                                                                    t5xxl_scaled_fp8 = t5xxl_scaled_fp8)
        else:
            raise ValueError(f"Invalid CLIP type: {clip_type}")

        # calculate the total number of parameters
        number_of_parameters = 0
        for _state_dict in state_dicts:
            number_of_parameters += comfy.utils.calculate_parameters(_state_dict)

        # create a dictionary with standard ComfyUI CLIP options
        model_options = { }
        if initial_device:  model_options["initial_device"] = initial_device
        if load_device   :  model_options["load_device"   ] = load_device
        if offload_device:  model_options["offload_device"] = offload_device

        # create a new CLIP instance from the components
        clip = cls.from_components(
            comfy_model_class       = comfy_model_class,
            comfy_tokenizer_class   = comfy_tokenizer_class,
            number_of_parameters    = number_of_parameters,
            comfy_tokenizer_options = {},
            comfy_model_options     = {},
            model_options           = model_options,
            )

        # load the weights into the new CLIP instance
        for _state_dict in state_dicts:
            missing_keys, unexpected_keys = clip.load_sd(_state_dict)
            if missing_keys:
                logging.warning(f"clip missing: {missing_keys}")
            if unexpected_keys:
                logging.debug(f"clip unexpected: {unexpected_keys}")

        return clip




