"""
File    : combined_autoencoder_model.py
Purpose : A class that combines two autoencoders into one model.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 13, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch.nn as nn

class CombinedAutoencoderModel(nn.Module):

    def __init__(self, autoencoder1: nn.Module, autoencoder2: nn.Module):
        super().__init__()
        self.autoencoder1 = autoencoder1
        self.autoencoder2 = autoencoder2

    def encode(self, x):
        output = self.autoencoder1.encode(x)
        if output is None:
            output = self.autoencoder2.encode(x)
        return output

    def decode(self, x):
        output = self.autoencoder1.decode(x)
        if output is None:
            output = self.autoencoder2.decode(x)
        return output

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
