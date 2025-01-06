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

    def normalize_state_dict(self, state_dict: dict) -> dict:
        return state_dict


# The list of supported formats for conversion.
# Each element has a `normalize_state_dict()` function that takes a `state_dict`
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
            supported_formats: A list of supported formats for the state dictionary.
                               Each format is a callable that takes a state dictionary and
                               a prefix, and returns a tuple containing the modified state
                               dictionary and the detected prefix. Default is _SUPPORTED_FORMATS.
        """

        # always ensure that `prefix` is normalized
        prefix = _normalize_prefix(prefix)

        # auto-detect prefix in the special case when prefix is set to "*"
        if prefix == "*":
            prefix = cls.detect_prefix(state_dict, default="")

        # normalize the state dictionary using the provided format converters
        state_dict, prefix = cls.normalize_state_dict(state_dict, prefix, supported_formats)

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
        if self.encoder:
            return self.encoder.device
        elif self.decoder:
            return self.decoder.device
        else:
            return None # torch.device("cpu")


    @device.setter
    def device(self, device):
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
    def normalize_state_dict(state_dict       : dict,
                             prefix           : str  = "",
                             supported_formats: list = _SUPPORTED_FORMATS
                             ) -> tuple[dict, str]:
        """
        Normalizes a state dictionary to match the native format.

        Args:
           state_dict       : A dictionary containing the model's state.
           prefix           : An optional prefix used to filter the state dictionary keys.
           supported_formats: An optional list of supported formats to try when normalizing.
                              (this parameter normally does not need to be provided)
        Returns:
           A tuple containing:
           - A dictionary containing the model's state in native format.
           - The new prefix to use with the new normalized dictionary.

        """
        assert prefix != "*", \
            f"AutoencoderModelEx: Prefix cannot be '*' when normalizing model tensors. Please provide a valid prefix."

        # always ensure that `prefix` is normalized
        prefix = _normalize_prefix(prefix)

        # remove the prefix from `state_dict`
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}
            prefix     = ""

        # iterate over all supported formats to find the appropriate one that can normalize the state_dict
        # (supported formats are usually `_PixArtFormat` and `_DiffusersFormat`)
        for format in supported_formats:

            # check whether this is the correct format to normalize `state_dict`
            if f"{prefix}{format.SIGNATURE_TENSORS[0]}" not in state_dict or \
               f"{prefix}{format.SIGNATURE_TENSORS[1]}" not in state_dict:
                continue

            # normalize `state_dict` using this supported format
            state_dict = format.normalize_state_dict(state_dict)

        return state_dict, prefix


    @staticmethod
    def infer_model_config(state_dict: dict,
                           prefix    : str = "",
                           ) -> dict:
        """
        Infers the model configuration from the state dictionary.

        Args:
           state_dict: A dictionary containing the model's state in native format.
           prefix    : An optional prefix used to filter the state dictionary keys.
        Returns:
           A dictionary containing the model's configuration.

        """
        assert prefix != "*", \
            f"AutoencoderModelEx: Prefix cannot be '*' when inferring model config. Please provide a valid prefix."

        # always ensure that `prefix` is normalized
        prefix = _normalize_prefix(prefix)

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
