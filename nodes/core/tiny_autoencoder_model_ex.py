"""
File    : tiny_autoencoder_model_ex.py
Purpose : Extension of the `TinyAutoencoderModel` class including additional functionality.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 10, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .tiny_autoencoder_model import TinyAutoencoderModel

# scale_factor/shift_factor values used for the latent space of different models
_SCALE_AND_SHIFT_BY_LATENT_FORMAT = {
    "sd"  : (0.18215, 0.    ),
    "sdxl": (0.13025, 0.    ),
    "sd3" : (1.5305 , 0.0609),
    "f1"  : (0.3611 , 0.1159)
}

def _normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix by ensuring it ends with a dot."""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


def _replace_prefix(state_dict: dict, old_prefix: str, new_prefix: str) -> dict:
    """Replaces the prefix of each key in a state dictionary."""
    fixed_dict: dict[str, torch.Tensor] = {}
    for key in state_dict.keys():
        if key.startswith(old_prefix):
            fixed_dict[key.replace(old_prefix, new_prefix,1)] = state_dict[key]
        else:
            fixed_dict[key] = state_dict[key]
    return fixed_dict


def _shift_layers(state_dict  : dict,
                  layer_prefix: str,
                  offset      : int
                  ) -> dict:
    """
    Returns a new state dictionary where layers are shifted by a specified offset.
    Args:
        state_dict    (dict): The original model parameters.
        layer_prefix   (str): The prefix used to identify the layers to shift.
        offset         (int): The number of layers to shift. Positive values shift the
                              layers forward, negative values shift them backward.
    """
    fixed_dict = {}
    for key, tensor in state_dict.items():

        if not key.startswith(layer_prefix):
            fixed_dict[key] = tensor
            continue

        _parts = key[len(layer_prefix):].split('.',1)
        if not _parts[0].isdecimal():
            fixed_dict[key] = tensor
            continue

        new_layer_number = int(_parts[0]) + offset
        dot_suffix       = f".{_parts[1]}" if len(_parts)>1 else ""
        fixed_key        = f"{layer_prefix}{new_layer_number}{dot_suffix}"
        fixed_dict[fixed_key] = tensor

    return fixed_dict


#------------------ HELPER FUNCTIONS FOR MODEL DETECTION -------------------#

def _detect_prefix(state_dict: dict, tensor_names: tuple) -> str | None:
    """Detects the common prefix shared by two tensors in a given state dictionary."""
    name = tensor_names[0]
    for key in state_dict.keys():
        if key.endswith(name):                                # check for the first signature tensor
            detected_prefix = key[:-len(name)]
            if detected_prefix+tensor_names[1] in state_dict: # check for the second signature tensor
                return detected_prefix
    return None


def _verify_tensors(state_dict: dict, prefix: str, tensor_names: tuple) -> bool:
    """Verifies the presence of two tensors based on a given prefix and tensor names."""
    return prefix+tensor_names[0] in state_dict and prefix+tensor_names[1] in state_dict


#------------------- SUPPORT FOR DIFFERENT MODEL FORMATS -------------------#

class _NativeFormat:
    """
    Identify tiny autoencoder models in its native format.
    """
    # these constants contain names of tensors that are characteristic of this format
    # and they are used to verify whether a checkpoint is compatible with the format.
    SIGNATURE_ENCODER_TENSORS = (
        "encoder.0.weight",
        "encoder.3.conv.0.bias"
    )
    SIGNATURE_DECODER_TENSORS = (
        "decoder.1.weight",
        "decoder.3.conv.0.bias"
    )
    def build_native_state_dict(self, state_dict: dict) -> dict:
        return state_dict


class _ComfyUIFormat:
    """
    Identify and translate tiny autoencoder models from the ComfyUI format.
    """
    SIGNATURE_ENCODER_TENSORS = (
        "taesd_encoder.0.weight",
        "taesd_encoder.3.conv.0.bias"
    )
    SIGNATURE_DECODER_TENSORS = (
        "taesd_decoder.1.weight",
        "taesd_decoder.3.conv.0.bias"
    )
    def build_native_state_dict(self, state_dict: dict) -> dict:
        state_dict = _replace_prefix(state_dict, "taesd_encoder.", "encoder.")
        state_dict = _replace_prefix(state_dict, "taesd_decoder.", "decoder.")
        return state_dict


class _HF_DiffusersFormat:
    """
    Identify and translate tiny autoencoder models from the HuggingFace Diffusers format.
    """
    SIGNATURE_ENCODER_TENSORS = (
        "encoder.layers.0.weight",
        "encoder.layers.1.conv.2.bias"
    )
    SIGNATURE_DECODER_TENSORS = (
        "decoder.layers.3.conv.0.weight",
        "decoder.layers.4.conv.2.bias"
    )
    def build_native_state_dict(self, state_dict: dict) -> dict:
        state_dict = _replace_prefix(state_dict, "encoder.layers.", "encoder.")
        state_dict = _replace_prefix(state_dict, "decoder.layers.", "decoder.")
        return state_dict


# The list of supported formats.
# Each element has a `build_native_state_dict()` function that takes a `state_dict`
# as input and returns a new `state_dict` with keys and tensors in native format.
_SUPPORTED_FORMATS = (_HF_DiffusersFormat(), _ComfyUIFormat(), _NativeFormat(), )


#===========================================================================#
#//////////////////////// TINY AUTOENCODER MODEL EX ////////////////////////#
#===========================================================================#
class TinyAutoencoderModelEx(TinyAutoencoderModel):

    @classmethod
    def from_state_dict(cls,
                        state_dict       : dict,
                        prefix           : str  = "",
                        *,
                        config           : dict = None,
                        filename         : str  = "",
                        return_config    : bool = False,
                        supported_formats: list = _SUPPORTED_FORMATS,
                        ) -> "TinyAutoencoderModelEx":
        """
        Creates a TinyAutoencoderModelEx instance from a state dictionary.

        Args:
            state_dict: A dictionary containing the model's state. The keys represent
                        the parameter names, and the values are the corresponding tensors.
            prefix    : An optional prefix used to filter the state dictionary keys.
                        If an asterisk "*" is specified, the prefix will be automatically detected.
            config    : A dictionary containing the model's configuration.
                        If None, the configuration is inferred from the state dictionary.
            return_config    : A boolean indicating whether to return the configuration along with the model.
            supported_formats: An optional list of supported formats to convert state_dict to native format.
                               (this parameter normally does not need to be provided)
        """

        # convert state_dict to native format using the provided format converters
        state_dict = cls.build_native_state_dict(state_dict, prefix,
                                                 filename          = filename,
                                                 supported_formats = supported_formats)

        # if no config was provided then try to infer it automatically from the keys of the state_dict
        if not config:
            config = cls.infer_model_config(state_dict)

        model = cls( **config )
        model.load_state_dict(state_dict, strict=False, assign=False)
        if return_config:
            return model, config
        return model


    @property
    def dtype(self):
        """Returns the data type of the model parameters."""
        return self.encoder.dtype if self.encoder else self.decoder.dtype


    @property
    def device(self):
        """Returns the device on which the model parameters are located."""
        return self.encoder.device if self.encoder else self.decoder.device


    @device.setter
    def device(self, device):
        """Dummy function for applications that try to set the `device` property (e.g. ComfyUI)"""
        pass


    def freeze(self) -> None:
        """Freeze all parameters of the model to prevent them from being updated during inference."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


    def unfreeze(self) -> None:
        """Unfreeze all parameters of the model to allow them to be updated during training."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()


    #__ useful static methods _______________________________________

    @staticmethod
    def detect_prefix(state_dict       : dict,
                      prefix           : str  = "*",
                      default          : str  = None,
                      supported_formats: list = _SUPPORTED_FORMATS
                      ) -> str:
        """
        Detects the prefix used by a given state dictionary.

        Args:
          state_dict       : A dictionary containing the model's state to detect.
          prefix           : An optional tentative prefix where detecting will be performed.
          supported_formats: An optional list of supported formats to try when detecting.
                             (this parameter normally does not need to be provided)
        Returns:
          The detected prefix used by the given state dictionary.
          (if no prefix is detected, it returns `None`)

        """
        prefix = _normalize_prefix(prefix)
        for format in supported_formats:

            # use the default auto-detection mechanism "*"
            if prefix == "*":
                detected_prefix = _detect_prefix(state_dict, format.SIGNATURE_ENCODER_TENSORS)
                if detected_prefix is not None:
                    return detected_prefix
                detected_prefix = _detect_prefix(state_dict, format.SIGNATURE_DECODER_TENSORS)
                if detected_prefix is not None:
                    return detected_prefix

            # if a tentative prefix was provided, use it to verify that it is valid
            elif _verify_tensors(state_dict, prefix, format.SIGNATURE_ENCODER_TENSORS):
                return prefix
            elif _verify_tensors(state_dict, prefix, format.SIGNATURE_DECODER_TENSORS):
                return prefix

        # if no prefix was detected then return `None` or the provided default
        return default



    @staticmethod
    def build_native_state_dict(state_dict       : dict,
                                prefix           : str  = "",
                                *,
                                filename         : str  = "",
                                supported_formats: list = _SUPPORTED_FORMATS
                                ) -> dict:
        """
        Returns a state dictionary that matches the native format for this model.

        Args:
           state_dict       : A dictionary containing the model's state in any format.
           prefix           : An optional prefix used to filter the state dictionary keys.
                              If an asterisk "*" is specified, the prefix will be automatically detected.
           supported_formats: An optional list of supported formats to convert state_dict to native format.
                              (this parameter normally does not need to be provided)
        Returns:
           A dictionary containing the model's state in native format.
        """

        # normalize prefix, forcing auto-detection when it is "*"
        prefix = _normalize_prefix(prefix)
        if prefix == "*":
            prefix = TinyAutoencoderModelEx.detect_prefix(state_dict, default="")

        # remove prefix from tensor names
        if prefix:
            state_dict = {name[len(prefix):]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}

        # try to convert `state_dict` to native format
        for format in supported_formats:
            if _verify_tensors(state_dict, "", format.SIGNATURE_ENCODER_TENSORS):
                state_dict = format.build_native_state_dict(state_dict)
                break

            if _verify_tensors(state_dict, "", format.SIGNATURE_DECODER_TENSORS):
                state_dict = format.build_native_state_dict(state_dict)
                break

        # in this point, we should have a native format `state_dict`,
        # but it may be necessary to apply a last fix to match this expected format:
        #  Layer |     Tensor     |  Module
        # -------+----------------+---------------------------
        #    0   |        -       |  Clamp()
        #    1   | [64, ch, 3, 3] |  conv(latent_channels, 64)
        #    2   |        -       |  ReLU()
        #  ....  |      .....     |  ......
        #
        # if the tensor "decoder.0.weight" exists (which should not exist),
        # then shift all the decoder layers by 1
        if f"decoder.0.weight" in state_dict:
            state_dict = _shift_layers(state_dict, layer_prefix="decoder.", offset=1)


        # add vae_scale and vae_shift if they are missing
        # (vae_scale and vae_shift are used for standard vae emulation)
        if "vae_scale" not in state_dict:
            if "encoder.vae_scale" not in state_dict or "decoder.vae_scale" not in state_dict:
                lower_case_filename = filename.lower()
                if  "f1" in lower_case_filename or "flux" in lower_case_filename:
                    state_dict["vae_scale"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["f1"  ][0] ])
                    state_dict["vae_shift"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["f1"  ][1] ])
                elif "sd3" in lower_case_filename:
                    state_dict["vae_scale"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sd3" ][0] ])
                    state_dict["vae_shift"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sd3" ][1] ])
                elif "sdxl" in lower_case_filename:
                    state_dict["vae_scale"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sdxl"][0] ])
                    state_dict["vae_shift"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sdxl"][1] ])
                else:
                    state_dict["vae_scale"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sd"  ][0] ])
                    state_dict["vae_shift"] = torch.tensor([ _SCALE_AND_SHIFT_BY_LATENT_FORMAT["sd"  ][1] ])

        return state_dict


    @staticmethod
    def infer_model_config(state_dict: dict) -> dict:
        """
        Infers the model configuration from a native state dictionary.

        Args:
           state_dict: A dictionary containing the model's state in native format.
                       This dictionary can be created using the `build_native_state_dict` method.
        Returns:
           A dictionary containing the model's configuration.
        """
        config = {
            "image_channels"               :     3    ,
            "latent_channels"              :     4    ,
            "encoder_hidden_channels"      :    64    ,
            "encoder_convolutional_layers" :     3    ,
            "encoder_res_blocks_per_layer" :     3    ,
            "decoder_hidden_channels"      :    64    ,
            "decoder_convolutional_layers" :     3    ,
            "decoder_res_blocks_per_layer" :     3    ,
            "encoder_latent_format"        : "unknown",
            "decoder_latent_format"        : "unknown",
        }
        return config
