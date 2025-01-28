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
*   `image.batch_size` (int): The number of images that are generated in each batch (e.g., 1).

### Sampler Parameters

*   `base.prompt` (str): The prompt used for the base model.
*   `base.negative` (str): The negative prompt used for the base model.
*   `base.steps` (int): The total number of steps for the base model.
*   `base.steps_start` (int): The step at which the base model's sampler begins the diffusion process.
*   `base.steps_end` (int): The step at which the base model's sampler terminates the diffusion process.
*   `base.steps_nfactor` (int): A factor used to expand or reduce the number of steps for the base model.
*   `base.cfg` (float): The guidance scale used by the base model's sampler.
*   `base.noise_seed` (int): The seed used for generating initial noise used by the base model.
*   `base.sampler_name` (str): The name of the sampler used by the base model.
*   `base.scheduler` (str): The name of the scheduler used by the base model.

### Sampler Parameters (Refiner)

*   `refiner.prompt` (str): The prompt used for the refiner model.
*   `refiner.negative` (str): The negative prompt used for the refiner model.
*   `refiner.steps` (int): The total number of steps for the refiner model.
*   `refiner.steps_start` (int): The step at which the refiner model's sampler begins the diffusion process.
*   `refiner.steps_end` (int): The step at which the refiner model's sampler terminates the diffusion process.
*   `refiner.steps_nfactor` (int): A factor used to expand or reduce the number of steps for the refiner model.
*   `refiner.cfg` (float): The guidance scale used by the refiner model's sampler.
*   `refiner.noise_seed` (int): The seed used for generating initial noise used by the refiner model.
*   `refiner.sampler_name` (str): The name of the sampler used by the refiner model.
*   `refiner.scheduler` (str): The name of the scheduler used by the refiner model.

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
