"""
File    : t5_encoder_model_ex.py
Purpose : Extension of the `T5EncoderModel` class including additional functionality.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 14, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .t5_encoder_model import T5EncoderModel

#===========================================================================#
#/////////////////////////// T5 ENCODER MODEL EX ///////////////////////////#
#===========================================================================#
class T5EncoderModelEx(T5EncoderModel):


    @property
    def dtype(self):
        """
        Returns the data type of the model parameters.
        (assuming that all parameters have the same data type)
        """
        return self.block[0].layer[0].SelfAttention.q.weight.dtype


    @property
    def device(self):
        """
        Returns the device on which the model parameters are located.
        (assuming that all parameters are on the same device)
        """
        return self.block[0].layer[0].SelfAttention.q.weight.device


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
    def infer_model_config(state_dict: dict) -> dict:
        """
        Infers the model configuration from a native state dictionary.

        Args:
           state_dict: A dictionary containing the model's state in native format.
                       This dictionary can be created using the `build_native_state_dict` method.
        Returns:
           A dictionary containing the model's configuration.
        """
        # TODO: implement code to infer the model configuration from the state dictionary
        config = {
            "d_ff"                            : 10240 ,
            "d_kv"                            :    64 ,
            "d_model"                         :  4096 ,
            "dense_act_fn"                    : "gelu_pytorch_tanh",
            "layer_norm_epsilon"              :  1e-6 ,
            "model_type"                      :  "t5" ,
            "is_gated_act"                    :  True ,
            "num_heads"                       :    64 ,
            "num_layers"                      :    24 ,
            "vocab_size"                      : 32128 ,
            "relative_attention_num_buckets"  :    32 ,
            "relative_attention_max_distance" :   128 ,
            "dropout_rate"                    :   0.1 ,
        }
        return config
