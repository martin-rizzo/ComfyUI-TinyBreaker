# GenParams

**GenParams** is a dictionary containing parameters used during the generation process, along with supplementary information.

## Important Notes

All parameters within **GenParams** are optional.  
The code should gracefully handle cases where parameters are missing by:
*   Assigning default values.
*   Performing additional calculations to predict the missing parameter's value.

## Parameters

### File Parameters

*   `file.name` (str): The name of the file containing the main model parameters (only the name and extension, without path)
*   `file.date` (str): The date when the file was generated, in ISO 8601 format (e.g., YYYY-MM-DD).

### Model Parameters

*   `modelspec.architecture` (str): The name of the architecture used by the model.
*   `modelspec.title` (str): The title of the model, used for identification.
*   `modelspec.date` (str): The date the model was created, in ISO 8601 format (e.g., YYYY-MM-DD).
*   `modelspec.license` (str): The license under which the model is distributed.
*   `modelspec.resolution` (str): The intended resolution for the model, formatted as `<width>x<height>` (e.g., "10240x1024").

### User-Provided Parameters

*   `user.style` (str): The name of the style selected by the user in the UI.
*   `user.prompt` (str): The prompt entered by the user in the UI.
*   `user.negative` (str): The negative prompt entered by the user in the UI.

### Image Parameters

*   `image.scale` (float): The scaling factor applied to the image size, based in the original resolution of the model (e.g., 1.22).
*   `image.orientation` (str): The orientation of the image (e.g., "landscape", "portrait").
*   `image.aspect_ratio` (str): The aspect ratio of the image (e.g., "16:9").
*   `image.upscale_factor` (float): The factor by which the image is upscaled (e.g., 2.5).
*   `image.batch_size` (int): The number of images that are generated in each batch (e.g., 1).

### Denoising Parameters

*   `denoising.base.prompt` (str): The prompt used for the base model.
*   `denoising.base.negative` (str): The negative prompt used for the base model.
*   `denoising.base.steps` (int): The total number of steps for the base model.
*   `denoising.base.steps_start` (int): The step at which the base model's sampler begins the diffusion process.
*   `denoising.base.steps_end` (int): The step at which the base model's sampler terminates the diffusion process.
*   `denoising.base.steps_nfactor` (int): A factor used to expand or reduce the number of steps for the base model.
*   `denoising.base.cfg` (float): The guidance scale used by the base model's sampler.
*   `denoising.base.noise_seed` (int): The seed used for generating initial noise used by the base model.
*   `denoising.base.sampler` (str): The name of the sampler used by the base model.
*   `denoising.base.scheduler` (str): The name of the scheduler used by the base model.

### Denoising Parameters (Refiner)

*   `denoising.refiner.prompt` (str): The prompt used for the refiner model.
*   `denoising.refiner.negative` (str): The negative prompt used for the refiner model.
*   `denoising.refiner.steps` (int): The total number of steps for the refiner model.
*   `denoising.refiner.steps_start` (int): The step at which the refiner model's sampler begins the diffusion process.
*   `denoising.refiner.steps_end` (int): The step at which the refiner model's sampler terminates the diffusion process.
*   `denoising.refiner.steps_nfactor` (int): A factor used to expand or reduce the number of steps for the refiner model.
*   `denoising.refiner.cfg` (float): The guidance scale used by the refiner model's sampler.
*   `denoising.refiner.noise_seed` (int): The seed used for generating initial noise used by the refiner model.
*   `denoising.refiner.sampler` (str): The name of the sampler used by the refiner model.
*   `denoising.refiner.scheduler` (str): The name of the scheduler used by the refiner model.

### Denoising Parameters (Upscaler)

*   `denoising.upscaler.prompt` (str): The prompt used for the model during upscaling.
*   `denoising.upscaler.negative` (str): The negative prompt used for the model during upscaling.
*   `denoising.upscaler.steps` (int): The total number of denoising steps during upscaling.
*   `denoising.upscaler.steps_start` (int): The step at which starting denoising for upscaling.
*   `denoising.upscaler.steps_end` (int): The step at which ending denoising for upscaling.
*   `denoising.upscaler.steps_nfactor` (int): A factor used to expand or reduce the number of denoising steps.
*   `denoising.upscaler.cfg` (float): The guidance scale used during upscaling.
*   `denoising.upscaler.noise_seed` (int): The seed used for generating initial noise during upscaling.
*   `denoising.upscaler.sampler` (str): The name of the sampler used during upscaling.
*   `denoising.upscaler.scheduler` (str): The name of the scheduler used during upscaling.

### Style Parameters

*   `styles.<name>.base.*`:
    *   `styles.<name>.base.prompt` (str): The base prompt for a specific style named `<name>`.
    *   `styles.<name>.base.negative` (str): The base negative prompt for a specific style named `<name>`.
    *   `...`
*   `styles.<name>.refiner.*`:
    *   `styles.<name>.refiner.prompt` (str): The refiner prompt for a specific style named `<name>`.
    *   `styles.<name>.refiner.negative` (str): The refiner negative prompt for a specific style named `<name>`.
    *   `...` (Indicates there may be additional refiner style-related parameters)

## Final Notes

* **GenParams** is used to pass information between different parts of the code during the generation process. It contains both user-provided and automatically calculated values.
* As a dictionary, it can store any parameter that needs to be passed around, but the parameters mentioned here are the most important used in the project.
