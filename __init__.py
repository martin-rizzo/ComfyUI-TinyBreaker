"""
File    : __init__.py
Purpose : Register the nodes of the ComfyUI-TinyBreaker project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)

    Copyright (c) 2024-2025 Martin Rizzo

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os

# initialize the TinyBreaker logger
from comfy.cli_args     import args
from .nodes.core.system import setup_logger
if os.getenv('TB_DEBUG'):
    setup_logger(log_level="DEBUG", use_stdout=args.log_stdout)
else:
    setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

# import the newly initialized TinyBreaker logger
from .nodes.core.system import logger

# initialize variables used by ComfyUI to import the custom nodes
WEB_DIRECTORY              = "./web"
NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


#================= TINYBREAKER CUSTOM NODE IMPORT PROCESS ==================#

_PROJECT_ID    = "//TinyBreaker"
_PROJECT_EMOJI = "💪"
_CATEGORY      = "TinyBreaker"
_DEPRECATED    = False

def _comfy_import_node(cls):
    global NODE_CLASS_MAPPINGS
    global NODE_DISPLAY_NAME_MAPPINGS

    class_name         = cls.__name__
    class_display_name = cls.TITLE
    class_category     = f"{_PROJECT_EMOJI}{_CATEGORY}"
    comfy_class_name   = f"{class_name} {_PROJECT_ID}"

    if class_name in NODE_CLASS_MAPPINGS:
        logger.warning(f"Node class {class_name} already exists, skipping import.")
        return

    if _DEPRECATED:
        cls.DEPRECATED = True
        class_display_name = class_display_name.replace("💪TB","")
        class_display_name = class_display_name.replace("| ","")
        class_display_name = f"❌{class_display_name} [Deprecated]"

    cls.CATEGORY = class_category
    NODE_CLASS_MAPPINGS[comfy_class_name]        = cls
    NODE_DISPLAY_NAME_MAPPINGS[comfy_class_name] = class_display_name


# TinyBreaker
_CATEGORY = "TinyBreaker"

from .nodes.tiny_dual_sampler                       import TinyDualSampler
_comfy_import_node(TinyDualSampler)

from .nodes.save_image                              import SaveImage
_comfy_import_node(SaveImage)

from .nodes.multiline_text                          import MutilineText
_comfy_import_node(MutilineText)



# TinyBreaker/genparams
_CATEGORY = "TinyBreaker/genparams"

from .nodes.select_style                            import SelectStyle
_comfy_import_node(SelectStyle)

from .nodes.set_base_seed                           import SetBaseSeed
_comfy_import_node(SetBaseSeed)

from .nodes.set_image_v2                            import SetImageV2
_comfy_import_node(SetImageV2)

from .nodes.set_image_details_v2                    import SetImageDetailsV2
_comfy_import_node(SetImageDetailsV2)

from .nodes.unified_prompt_input                    import UnifiedPromptInput
_comfy_import_node(UnifiedPromptInput)

from .nodes.unpack_boolean                          import UnpackBoolean
_comfy_import_node(UnpackBoolean)

from .nodes.unpack_sampler_params                   import UpackSamplerParams
_comfy_import_node(UpackSamplerParams)



# TinyBreakers/loaders
_CATEGORY = "TinyBreaker/loaders"

from .nodes.load_tinybreaker_ckpt                   import LoadTinyBreakerCkpt
_comfy_import_node(LoadTinyBreakerCkpt)

from .nodes.load_tinybreaker_ckpt_advanced          import LoadTinyBreakerCkptAdvanced
_comfy_import_node(LoadTinyBreakerCkptAdvanced)

from .nodes.load_t5_encoder_experimental            import LoadT5EncoderExperimental
_comfy_import_node(LoadT5EncoderExperimental)

from .nodes.load_any_vae                            import LoadAnyVAE
_comfy_import_node(LoadAnyVAE)

from .nodes.load_partial_vae                        import LoadPartialVAE
_comfy_import_node(LoadPartialVAE)



# TinyBreaker/latent
_CATEGORY = "TinyBreaker/latent"

from .nodes.empty_latent_image                      import EmptyLatentImage
_comfy_import_node(EmptyLatentImage)

from .nodes.tiny_encode                             import TinyEncode
_comfy_import_node(TinyEncode)

from .nodes.tiny_decode                             import TinyDecode
_comfy_import_node(TinyDecode)



# TinyBreaker/transcoding
_CATEGORY = "TinyBreaker/transcoding"

from .nodes.load_transcoder                         import LoadTranscoder
_comfy_import_node(LoadTranscoder)

from .nodes.build_custom_transcoder                 import BuildCustomTranscoder
_comfy_import_node(BuildCustomTranscoder)

from .nodes.transcode_latent                        import TranscodeLatent
_comfy_import_node(TranscodeLatent)

from .nodes.transcode_latent_two_steps              import TranscodeLatentTwoSteps
_comfy_import_node(TranscodeLatentTwoSteps)



# TinyBreaker/upscaler
_CATEGORY = "TinyBreaker/upscaler"

from .nodes.tiny_upscaler                           import TinyUpscaler
_comfy_import_node(TinyUpscaler)

from .nodes.tiny_upscaler_advanced                  import TinyUpscalerAdvanced
_comfy_import_node(TinyUpscalerAdvanced)



# TinyBreaker/__dev
_CATEGORY = "TinyBreaker/__dev"

from .nodes.genparams_debug_logger                  import GenParamsDebugLogger
_comfy_import_node(GenParamsDebugLogger)

from .nodes.save_anything                           import SaveAnything
_comfy_import_node(SaveAnything)



# TinyBreaker/__deprecated
_CATEGORY   = "TinyBreaker/__deprecated"
_DEPRECATED = True

from .nodes.deprecated_nodes.load_tinybreaker_checkpoint        import LoadTinyBreakerCheckpoint
_comfy_import_node(LoadTinyBreakerCheckpoint)

from .nodes.deprecated_nodes.load_tinybreaker_checkpoint_custom import LoadTinyBreakerCheckpointCustom
_comfy_import_node(LoadTinyBreakerCheckpointCustom)

from .nodes.deprecated_nodes.double_stage_sampler               import DoubleStageSampler
_comfy_import_node(DoubleStageSampler)

from .nodes.deprecated_nodes.set_float                          import SetFloat
_comfy_import_node(SetFloat)

from .nodes.deprecated_nodes.set_image                          import SetImage
_comfy_import_node(SetImage)

from .nodes.deprecated_nodes.set_image_tweaks                   import SetImageTweaks
_comfy_import_node(SetImageTweaks)



logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")

