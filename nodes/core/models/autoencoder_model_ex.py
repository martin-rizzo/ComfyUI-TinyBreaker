"""
File    : autoencoder_model_ex.py
Purpose : Extension of the AutoencoderModel class to include additional useful funtionality.
          (includes autodetection of model comfiguation and support for HF diffusers format)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 5, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .autoencoder_model import AutoencoderModel


def _normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix by ensuring it ends with a dot."""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


#------------------ HELPER FUNCTIONS FOR MODEL DETECTION -------------------#

def _get_tensor_number(key: str, prefix: str, *, default: int = 0) -> int:
    """Extracts a number from a key based on a given prefix. Returns 0 if not found."""
    number_str = key[len(prefix):].split('.',1)[0]
    return int(number_str) if number_str.isdigit() else default


def _get_max_tensor_number(state_dict: dict, prefix: str, *, default: int = 0) -> int:
    """Finds and returns the maximum number from keys in a state dictionary based on a given prefix. Returns 0 if not found."""
    return max([_get_tensor_number(key, prefix) for key in state_dict.keys() if key.startswith(prefix)], default=default)


def _get_tensor_size(state_dict: dict, name: str, *, dim: int, default: int = 0) -> int:
    """Returns the size of a tensor in a given dimension based on its name."""
    tensor = state_dict.get(name)
    return tensor.shape[dim] if tensor is not None else default


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
    Identify and process autoencoder models in its native format.
    """
    # these constants contain names of tensors that are characteristic of this format
    # and they are used to verify whether a checkpoint is compatible with the format.
    SIGNATURE_ENCODER_TENSORS = (
        "encoder.conv_in.weight",
        "encoder.down.0.block.0.conv1.weight"
    )
    SIGNATURE_DECODER_TENSORS = (
        "decoder.conv_in.weight",
        "decoder.up.0.block.0.conv1.weight"
    )
    def build_native_state_dict(self, state_dict: dict) -> dict:
        return state_dict


class _HF_DiffusersFormat:
    """
    Identify and process autoencoder models in HF diffusers format.
    ATTENTION: No support for HF diffusers format yet !!
    """
    def build_native_state_dict(self, state_dict: dict) -> dict:
        # TODO: add support for HF diffusers format in AutoencoderModelEx
        return None


# The list of supported formats.
# Each element has a `build_native_state_dict()` function that takes a `state_dict`
# as input and returns a new `state_dict` with keys and tensors in native format.
_SUPPORTED_FORMATS = (_NativeFormat(), )



#===========================================================================#
#////////////////////////// AUTOENCODER MODEL EX ///////////////////////////#
#===========================================================================#
class AutoencoderModelEx(AutoencoderModel):

    @classmethod
    def from_state_dict(cls,
                        state_dict       : dict,
                        prefix           : str  = "",
                        *,# keyword-only arguments #
                        config           : dict = None,
                        supported_formats: list = _SUPPORTED_FORMATS,
                        nn = None,
                        ) -> tuple[ "AutoencoderModelEx", dict, list, list ]:
        """
        Creates an AutoencoderModelEx instance from a state dictionary.

        Args:
            state_dict: A dictionary containing the model's state. The keys represent
                        the parameter names, and the values are the corresponding tensors.
            prefix    : An optional prefix used to filter the state dictionary keys.
                        If an asterisk "*" is specified, the prefix will be automatically detected.
            config    : A dictionary containing the model's configuration.
                        If None, the configuration is inferred from the state dictionary.
            supported_formats: An optional list of supported formats to convert state_dict to native format.
                               (this parameter normally does not need to be provided)
            nn (optional): The neural network module to use. Defaults to `torch.nn`.
                           This parameter allows for the injection of custom or
                           optimized implementations of "nn" modules.
        """

        # convert state_dict to native format using the provided format converters
        state_dict = cls.build_native_state_dict(state_dict, prefix,
                                                 supported_formats = supported_formats)

        # if no config was provided then try to infer it automatically from the state_dict
        if not config:
            config = cls.infer_model_config(state_dict)

        # if `nn` was provided then overwrite it in the config
        if nn is not None:
            config["nn"] = nn

        model = cls( **config )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=False)
        return model, config, missing_keys, unexpected_keys


    @property
    def dtype(self):
        """
        Returns the data type of the model parameters.
        (assuming that all parameters have the same data type)
        """
        return self.get_encoder_dtype() or self.get_decoder_dtype() or torch.float16


    @property
    def device(self):
        """
        Returns the device on which the model parameters are located.
        (assuming that all parameters are on the same device)
        """
        return self.get_encoder_device() or self.get_decoder_device() or torch.device("cpu")


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
            prefix = AutoencoderModelEx.detect_prefix(state_dict, default="")

        # remove prefix from tensor names
        if prefix:
            unpref_state_dict = {name[len(prefix):]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}
        else:
            unpref_state_dict = state_dict

        # generate the native `state_dict` using the format that matches the tensors
        for format in supported_formats:
            if _verify_tensors(unpref_state_dict, "", format.SIGNATURE_ENCODER_TENSORS):
                return format.build_native_state_dict(unpref_state_dict)
            if _verify_tensors(unpref_state_dict, "", format.SIGNATURE_DECODER_TENSORS):
                return format.build_native_state_dict(unpref_state_dict)

        # in case that it does not match any format, return the unprefixed `state_dict`
        return unpref_state_dict


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
        SIGNATURE_ENCODER_TENSORS = _NativeFormat.SIGNATURE_ENCODER_TENSORS
        SIGNATURE_DECODER_TENSORS = _NativeFormat.SIGNATURE_DECODER_TENSORS

        # currently, only models with double encoder channels are supported.
        # automatically detecting the correct relationship between encoder and decoder latent channel
        # counts can be difficult, as some of them (the encoder or decoder) may not be present.
        use_double_encoding_channels = True

        # starting with everything set to zero
        latent_channels              = 0
        image_channels               = 0
        encoder_hidden_channels      = 0
        encoder_channel_multipliers  = None
        encoder_res_blocks_per_layer = 0
        decoder_hidden_channels      = 0
        decoder_channel_multipliers  = None
        decoder_res_blocks_per_layer = 0

        # count the number of keys in each module to detect which one is present
        encoder_keys_count = sum(1 for name in SIGNATURE_ENCODER_TENSORS if name in state_dict)
        decoder_keys_count = sum(1 for name in SIGNATURE_DECODER_TENSORS if name in state_dict)

        # encoder auto-detection
        if encoder_keys_count > 0:
            latent_channels              = _get_tensor_size(state_dict, "encoder.conv_out.weight", dim=0, default=8)
            if use_double_encoding_channels:
                latent_channels = latent_channels // 2
            image_channels               = _get_tensor_size(state_dict, "encoder.conv_in.weight", dim=1, default=3)
            encoder_hidden_channels      = _get_tensor_size(state_dict, "encoder.conv_in.weight", dim=0, default=128)
            encoder_channel_multipliers  = [1, 2, 4, 4]
            encoder_res_blocks_per_layer = 1 + _get_max_tensor_number(state_dict, "encoder.down.0.block.", default=1)

        # decoder auto-detection
        if decoder_keys_count > 0:
            latent_channels              = _get_tensor_size(state_dict, "decoder.conv_in.weight" , dim=1, default=4)
            image_channels               = _get_tensor_size(state_dict, "decoder.conv_out.weight", dim=0, default=3)
            decoder_hidden_channels      = _get_tensor_size(state_dict, "decoder.conv_out.weight", dim=1, default=128)
            decoder_channel_multipliers  = [1, 2, 4, 4]
            decoder_res_blocks_per_layer = _get_max_tensor_number(state_dict, "decoder.up.0.block.", default=2)


        config = {
            "image_channels"              :   image_channels,
            "latent_channels"             :   latent_channels,
            "pre_quant_channels"          :   latent_channels,
            "encoder_hidden_channels"     : encoder_hidden_channels,
            "encoder_channel_multipliers" : encoder_channel_multipliers,
            "encoder_res_blocks_per_layer": encoder_res_blocks_per_layer,
            "decoder_hidden_channels"     : decoder_hidden_channels,
            "decoder_channel_multipliers" : decoder_channel_multipliers,
            "decoder_res_blocks_per_layer": decoder_res_blocks_per_layer,
            "use_deterministic_encoding"  :   True,
            "use_double_encoding_channels": use_double_encoding_channels,
        }
        return config
