# What is TinyBreaker?

**TinyBreaker** is a hybrid model that combines the [PixArt model](https://github.com/PixArt-alpha/PixArt-sigma) for base image generation with Photon model (or any SD1 model) for image refinement. The idea is to leverage both models' strengths in these tasks, enabling them to operate efficiently on mid and low-end hardware due to their minimal parameter count. Moreover, by sequentially executing both models, you can offload them to system RAM reducing the VRAM usage. Additionally, TinyBreaker employs [Tiny Autoencoders](https://github.com/madebyollin/taesd) for latent space conversion, optimizing performance and efficiency.

**TinyBreaker** is the natural evolution of my two previous developments:
- **[Photon Model](https://civitai.com/models/84728/photon)**: A fine-tuning of SD1.5 aimed at generating photorealistic and visually appealing images effortlessly.
- **[The Abominable Workflows](https://civitai.com/models/420163)**: A set of workflows for ComfyUI that emulated, through a spaghetti nightmare, what TinyBreaker currently achieves.


## What's Great About TinyBreaker?

**Efficient Parameter Use:**
TinyBreaker is notable for its low parameter count, featuring just 0.6 billion parameters in the base model. This efficiency means that high-quality image generation requires significantly fewer computational resources compared to heavier models.

**Quick Performance:**
Currently, TinyBreaker can produce an image of size 1536×1024 within approximately 10 to 15 seconds using an NVIDIA RTX 3080 GPU. Efforts are ongoing to enhance the model's speed further, aiming for even faster performance without sacrificing quality.

**High Prompt Adherence:**
Thanks to the PixArt model integration, TinyBreaker achieves impressive adherence to prompts despite its minimal parameter count. This ensures that the generated images closely align with user instructions and expectations.


## The Technical Edge

**Hybrid Architecture:**
TinyBreaker's design leverages the PixArt model for creating a strong base image and uses either a Photon or SD1 model to refine these images. By combining their strengths, the model achieve high-quality results while minimizing computational demands.

**Optimized Latent Space Handling:**
The use of Tiny Autoencoders for latent space conversion further boosts TinyBreaker's performance and efficiency. These autoencoders streamline the process of converting input data into meaningful images, ensuring quality while minimizing resource usage.

## Future Directions
My focus is on enhancing speed without compromising image quality, and I'm working hard to make TinyBreaker more accessible, especially for those with mid-range or lower-end hardware.

Thank you for your continued support as we strive to improve. Updates will be shared soon, looking forward to sharing what's new!

