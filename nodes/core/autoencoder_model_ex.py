"""
File    : autoencoder_model_ex.py
Purpose : Extension of the Autoencoder class to include additional useful funtionality.
          (includes autodetection of model comfiguation and support for HF diffusers format)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 5, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .autoencoder_model import AutoencoderModel

def _normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix by ensuring it ends with a dot."""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


#------------------- SUPPORT FOR DIFFERENT MODEL FORMATS -------------------#

class _NativeFormat:

    # format must have its `SIGNATURE_TENSORS` constant with
    # the names of tensors that are characteristic to it
    SIGNATURE_TENSORS = [
        "decoder.conv_in.weight",
        "decoder.up.0.block.0.nin_shortcut.weight"
    ]

    def build_native_state_dict(self, state_dict: dict) -> dict:
        return state_dict


# The list of supported formats for conversion.
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
                        config           : dict = None,
                        supported_formats: list = _SUPPORTED_FORMATS
                        ) -> "AutoencoderModelEx":
        """
        Creates a PixArtModel instance from a state dictionary.

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
        state_dict = cls.build_native_state_dict(state_dict, prefix, supported_formats)

        # if no config was provided then try to infer it automatically from the keys of the state_dict
        if not config:
            config = cls.infer_model_config(state_dict)

        model = cls( **config )
        model.load_state_dict(state_dict, strict=False, assign=False)
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


    #====== Useful Static Methods ======#

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
            name1 = format.SIGNATURE_TENSORS[0]
            name2 = format.SIGNATURE_TENSORS[1]

            # the default auto-detection mechanism
            if prefix == "*":
                for key in state_dict.keys():
                    if key.endswith(name1):                     # check for the first signature tensor
                        detected_prefix = key[:-len(name1)]
                        if detected_prefix+name2 in state_dict: # check for the second signature tensor
                            return detected_prefix

            # if a tentative prefix was provided, use it for detection
            elif prefix+name1 in state_dict and prefix+name2 in state_dict:
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
            if format.SIGNATURE_TENSORS[0] in unpref_state_dict and format.SIGNATURE_TENSORS[1] in unpref_state_dict:
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

        # TODO: implement config detection

        config = {
            "image_channels"              :     3,
            "latent_channels"             :     4,
            "pre_quant_channels"          :     4,
            "encoder_hidden_channels"     :   128,
            "encoder_channel_multipliers" : [1, 2, 4, 4],
            "encoder_res_blocks_per_layer":     2,
            "decoder_hidden_channels"     :   128,
            "decoder_channel_multipliers" : [1, 2, 4, 4],
            "decoder_res_blocks_per_level":     2,
            "use_deterministic_encoding"  :  True,
            "use_double_encoding_channels":  True,
        }
        return config
