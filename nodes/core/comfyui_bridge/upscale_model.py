"""
File    : comfyui_bridge/upscale_model.py
Purpose : The standard "UPSCALE_MODEL" transmitted through ComfyUI's node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Aug 29, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from   ..safetensors import filter_state_dict, normalize_prefix
from   spandrel      import ModelLoader, ImageModelDescriptor
import comfy.utils


class UpscaleModel(ImageModelDescriptor):
    """
    A wrapper for the spandrel's "ImageModelDescriptor" class.
    This is the native object that travels through ComfyUI's "UPSCALE_MODEL" threads.
    """

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        *,#-- keyword-only arguments --#
                        prefix: str = None,
                        ) -> "UpscaleModel":
        """
        Create an UpscaleModel instance from a state dictionary.

        Args:
            state_dict: Dictionary containing model weights and configuration
            prefix    : Optional prefix to filter keys in the state dictionary

        Returns:
            UpscaleModel instance if the state dictionary is valid, None if empty or invalid.
        """
        if state_dict is None:
            state_dict = {}

        # filter the state dict by prefix (if given)
        if prefix:
            prefix     = normalize_prefix(prefix)
            state_dict = filter_state_dict(state_dict, prefix)

        # if the resulting state dict is empty, then we can't load the model
        if len(state_dict) == 0:
            return None

        # remove "module." prefix for some models (?)
        # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.55/comfy_extras/nodes_upscale_model.py#L29
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
            state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.":""})

        # load the upscale model using spandrel's ModelLoader
        # https://github.com/chaiNNer-org/spandrel
        model = ModelLoader().load_from_state_dict(state_dict).eval()
        if not isinstance(model, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return model


