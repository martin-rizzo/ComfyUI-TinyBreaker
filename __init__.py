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
from .nodes.utils.system import logger
WEB_DIRECTORY              = "./web"
NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_PROJECT_ID    = "//TinyBreaker"
_PROJECT_EMOJI = "💪"
_CATEGORY      = "TinyBreaker"
_DEPRECATED    = False

def comfy_import_node(cls):
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
        class_display_name = class_display_name.replace("💪TB","")
        class_display_name = class_display_name.replace("| ","")
        class_display_name = f"❌{class_display_name} [Deprecated]"

    cls.CATEGORY = class_category
    NODE_CLASS_MAPPINGS[comfy_class_name]        = cls
    NODE_DISPLAY_NAME_MAPPINGS[comfy_class_name] = class_display_name



# TinyBreaker/genparams
_CATEGORY = "TinyBreaker/genparams"

from .nodes.select_style                            import SelectStyle
comfy_import_node(SelectStyle)

from .nodes.set_base_seed                           import SetBaseSeed
comfy_import_node(SetBaseSeed)

from .nodes.set_image_v2                            import SetImageV2
comfy_import_node(SetImageV2)

from .nodes.set_image_details_v2                    import SetImageDetailsV2
comfy_import_node(SetImageDetailsV2)

from .nodes.unified_prompt_input                    import UnifiedPromptInput
comfy_import_node(UnifiedPromptInput)

from .nodes.unpack_sampler_params                   import UpackSamplerParams
comfy_import_node(UpackSamplerParams)



# TinyBreakers/loaders
_CATEGORY = "TinyBreaker/loaders"

from .nodes.load_tinybreaker_checkpoint_v2          import LoadTinyBreakerCheckpointV2
comfy_import_node(LoadTinyBreakerCheckpointV2)

from .nodes.load_tinybreaker_checkpoint_custom_v2   import LoadTinyBreakerCheckpointCustomV2
comfy_import_node(LoadTinyBreakerCheckpointCustomV2)

from .nodes.load_t5_encoder_experimental            import LoadT5EncoderExperimental
comfy_import_node(LoadT5EncoderExperimental)

from .nodes.load_any_vae                            import LoadAnyVAE
comfy_import_node(LoadAnyVAE)

from .nodes.load_partial_vae                        import LoadPartialVAE
comfy_import_node(LoadPartialVAE)



# TinyBreakers/transcoding
_CATEGORY = "TinyBreaker/transcoding"

from .nodes.load_transcoder                         import LoadTranscoder
comfy_import_node(LoadTranscoder)

from .nodes.build_custom_transcoder                 import BuildCustomTranscoder
comfy_import_node(BuildCustomTranscoder)

from .nodes.transcode_latent                        import TranscodeLatent
comfy_import_node(TranscodeLatent)

from .nodes.transcode_latent_two_steps              import TranscodeLatentTwoSteps
comfy_import_node(TranscodeLatentTwoSteps)



# TinyBreaker
_CATEGORY = "TinyBreaker"

from .nodes.double_stage_sampler                    import DoubleStageSampler
comfy_import_node(DoubleStageSampler)

from .nodes.empty_latent_image                      import EmptyLatentImage
comfy_import_node(EmptyLatentImage)

from .nodes.save_image                              import SaveImage
comfy_import_node(SaveImage)

from .nodes.tiny_upscale_prototype1                 import TinyUpscalePrototype1
comfy_import_node(TinyUpscalePrototype1)


# TinyBreaker/__dev
_CATEGORY = "TinyBreaker/__dev"

from .nodes.genparams_debug_logger                  import GenParamsDebugLogger
comfy_import_node(GenParamsDebugLogger)

from .nodes.save_anything                           import SaveAnything
comfy_import_node(SaveAnything)



# TinyBreaker/__deprecated
_CATEGORY   = "TinyBreaker/__deprecated"
_DEPRECATED = True

from .nodes.deprecated.load_tinybreaker_checkpoint          import LoadTinyBreakerCheckpoint
comfy_import_node(LoadTinyBreakerCheckpoint)

from .nodes.deprecated.load_tinybreaker_checkpoint_custom   import LoadTinyBreakerCheckpointCustom
comfy_import_node(LoadTinyBreakerCheckpointCustom)

from .nodes.deprecated.set_float                            import SetFloat
comfy_import_node(SetFloat)

from .nodes.deprecated.set_image                            import SetImage
comfy_import_node(SetImage)

from .nodes.deprecated.set_image_tweaks                     import SetImageTweaks
comfy_import_node(SetImageTweaks)



logger.info(f"Imported {len(NODE_CLASS_MAPPINGS)} nodes")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
