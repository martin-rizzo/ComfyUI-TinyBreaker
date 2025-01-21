# GenParams

**GenParams** is a dictionary containing parameters used during the generation process, along with supplementary information.

## Important Notes

All parameters within **GenParams** are optional. The code should gracefully handle cases where parameters are missing by:

*   Assigning default values.
*   Performing additional calculations to predict the missing parameter's value.

## Parameters

### Model Parameters

*   `filename` (str): The name of the file containing the model parameters (only the name and extension, without path)
*   `modelspec.architecture` (str): The name of the architecture used by the model.
*   `modelspec.title` (str): The title of the model, used for identification.
*   `modelspec.date` (str): The date the model was created, in ISO 8601 format (e.g., YYYY-MM-DD).
*   `modelspec.license` (str): The license under which the model is distributed.
*   `modelspec.resolution` (str): The intended resolution for the model, formatted as `<width>x<height>` (e.g., "1920x1080").

### User-Provided Parameters

*   `user.style` (str): The name of the style selected by the user in the UI.
*   `user.prompt` (str): The prompt entered by the user in the UI.
*   `user.negative` (str): The negative prompt entered by the user in the UI.

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



