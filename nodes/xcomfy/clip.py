"""
File    : xcomfy/clip.py
Purpose : The standard CLIP object transmitted through ComfyUI's node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 3, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.sd
import folder_paths

#===========================================================================#
#////////////////////////////////// CLIP ///////////////////////////////////#
#===========================================================================#

class CLIP(comfy.sd.CLIP):

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str = "",
                        type      : str = "stable_diffusion"
                        ) -> "CLIP":

        # ensure the prefix ends with a dot
        if prefix and not prefix.endswith("."):
            prefix += "."

        # get the directory where embeddings are stored
        embedding_directory = folder_paths.get_folder_paths("embeddings")

        # identify the CLIP type
        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        elif type == "mochi":
            clip_type = comfy.sd.CLIPType.MOCHI
        # elif type == "ltxv":
        #     clip_type = comfy.sd.CLIPType.LTXV
        # elif type == "pixart":
        #     clip_type = comfy.sd.CLIPType.PIXART
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        # try to remove `prefix` from the state dict keys
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}


        # load the CLIP model using the comfyui functionality
        clip = comfy.sd.load_text_encoder_state_dicts([state_dict],
                                                      embedding_directory = embedding_directory,
                                                      clip_type           = clip_type)
        return clip

