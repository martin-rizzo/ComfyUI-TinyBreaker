"""
File    : transcoder_model_ex.py
Purpose : Provides an extended version of the `TranscoderModel` class with
          features like automatic detection of model configuration. It is
          designed to be used alongside `transcoder_model.py` in any project
          with minimal external dependencies.
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
from .transcoder_model import TranscoderModel

def _normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix by ensuring it ends with a dot."""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


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
    Identify transcoders models in its native format.
    """
    # these constants contain names of tensors that are characteristic of this format
    # and they are used to verify whether a checkpoint is compatible with the format.
    SIGNATURE_TENSORS = (
        "transd.3.conv.0.bias",
        "transe.3.conv.0.weight"
    )
    def build_native_state_dict(self, state_dict: dict) -> dict:
        return state_dict


# The list of supported formats.
# Each element has a `build_native_state_dict()` function that takes a `state_dict`
# as input and returns a new `state_dict` with keys and tensors in native format.
_SUPPORTED_FORMATS = (_NativeFormat(), )



#===========================================================================#
#/////////////////////////// TRANSCODER MODEL EX ///////////////////////////#
#===========================================================================#
class TranscoderModelEx(TranscoderModel):
    """Extension of the TranscoderModel class with additional functionality."""

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str  = "",
                        *,# keyword-only arguments #
                        config           : dict = None,
                        supported_formats: list = _SUPPORTED_FORMATS,
                        ) -> tuple[ "TranscoderModelEx", dict, list, list ]:
        """
        Create a TranscoderModelEx instance from a state dictionary.

        Args:
            state_dict: A dictionary containing the model's state. The keys represent
                        the parameter names, and the values are the corresponding tensors.
            prefix    : An optional prefix used to filter the state dictionary keys.
                        If an asterisk "*" is specified, the prefix will be automatically detected.
            config    : A dictionary containing the model's configuration.
                        If None, the configuration is inferred from the state dictionary.
            supported_formats: An optional list of supported formats to convert state_dict to native format.
                                (this parameter normally does not need to be provided)
        """

        # convert state_dict to native format using the provided format converters
        state_dict = cls.build_native_state_dict(state_dict, prefix,
                                                 supported_formats = supported_formats)

        # if no config was provided then try to infer it automatically from the state_dict
        if not config:
            config = cls.infer_model_config(state_dict)

        model = cls( **config )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=False)
        return model, config, missing_keys, unexpected_keys


    @property
    def dtype(self):
        """Returns the data type of the model parameters."""
        return self.transe.dtype if self.transe else self.transd.dtype


    @property
    def device(self):
        """Returns the device on which the model parameters are located."""
        return self.transe.device if self.transe else self.transd.device


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
        self.unnormalized_adapter_in.freeze()
        self.unnormalized_adapter_out.freeze()
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
                detected_prefix = _detect_prefix(state_dict, format.SIGNATURE_TENSORS)
                if detected_prefix is not None:
                    return detected_prefix

            # if a tentative prefix was provided, use it to verify that it is valid
            elif _verify_tensors(state_dict, prefix, format.SIGNATURE_TENSORS):
                return prefix

        # if no prefix was detected then return `None` or the provided default
        return default


    @staticmethod
    def build_native_state_dict(state_dict: dict,
                                prefix    : str  = "",
                                *,# keyword-only arguments #
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
            prefix = TranscoderModelEx.detect_prefix(state_dict, default="")

        # remove prefix from tensor names
        if prefix:
            state_dict = {name[len(prefix):]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}

        # try to convert to native format
        for format in supported_formats:
            if _verify_tensors(state_dict, "", format.SIGNATURE_TENSORS):
                state_dict = format.build_native_state_dict(state_dict)
                break

        # return the un-prefixed, native state dictionary
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
        # TODO: implement this method to infer the model configuration from a state dictionary

        config = {
            "input_channels"                :   4 ,
            "output_channels"               :   4 ,
            "decoder_intermediate_channels" :  64 ,
            "decoder_convolutional_layers"  :   3 ,
            "decoder_res_blocks_per_layer"  :   3 ,
            "encoder_intermediate_channels" :  64 ,
            "encoder_convolutional_layers"  :   3 ,
            "encoder_res_blocks_per_layer"  :   3 ,
            "use_internal_rgb_layer"        : True,
            "use_gaussian_blur_bridge"      : True,
            "input_latent_format"           : "unknown",
            "output_latent_format"          : "unknown",
        }
        return config
