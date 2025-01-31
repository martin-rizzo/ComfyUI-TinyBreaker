# Unified Prompt Input Node

The **Unified Prompt Input** node allows you to generate images using a text description. This description includes the prompt itself, followed by a series of parameters. These parameters, identified by names starting with a double hyphen (--), modify the generated image. 

The following sections detail each available parameter.

## Minor Adjustments

These parameters allow for subtle modifications and fine-tuning of the generated image.

### `--no <elements>`

*   **Description:** Specifies elements that should **not** appear in the generated image. (negative prompt)
*   **Example:** `--no trees, sky, clouds`
*   **Usage:** Useful for removing unwanted elements from the final result.

### `--refine <description>`

*   **Description:** Provides a textual description of what elements should be refined in the image.
*   **Example:** `--refine trees and cat
*   **Usage:** Helps to refine specific elements in the image.

### `--variant <number>`

*   **Description:** Specifies variants of the refinement process without changing the composition of the image.
*   **Example:** `--variant 2`
*   **Usage:** Although the image remains practically unchanged, different variants can solve errors in hands and faces.
*   **Values:** An integer indicating the desired variant.

### `--cfg-adjust <value>`

*   **Description:** Adjusts the correction value to the default Classifier-Free Guidance (CFG).
*   **Example:** `--cfg-adjust -0.2`
*   **Usage:** Allows fine-tuning the effect of the Classifier-Free Guidance (CFG).
*   **Values:** A floating-point number. A value of `0.0` is the best value for the model.

### `--detail <level>`

*   **Description:** Controls the level of detail in the generated image.
*   **Example:** `--detail normal`
*   **Usage:** Allows adjusting the amount of detail and textures in the image.
*   **Options:**
    *   `none`: Disables the refinement process
    *   `minimal`: Minimal detail.
    *   `low`: Low level of detail.
    *   `normal`: Standard level of detail. (recommended)
    *   `high`: High level of detail.
    *   `veryhigh`: Very high level of detail.
    *   `maximum`: Maximum level of detail.

## Major Changes

These parameters allow for significant changes in the composition, structure and/or style of the generated image.

### `--seed <number>`

*   **Description:** Defines a number used to initialize the random generator.
*   **Example:** `--seed 42`
*   **Usage:** Different seeds produce completely different images.
*   **Values:** An integer number.

### `--aspect <ratio>`

*   **Description:** Specifies the aspect ratio of the image.
*   **Example:** `--aspect 16:9`
*   **Usage:** Defines the proportion between the width and height of the image.
*   **Options:**
    *   `1:1`
    *   `4:3`
    *   `3:2`
    *   `16:10`
    *   `16:9`
    *   `2:1`
    *   `21:9`
    *   `12:5`
    *   `70:27`
    *   `32:9`

### `--landscape` / `--portrait`

*   **Description:** Specifies the orientation of the image.
*   **Usage:**
    *   `--landscape`: Generates an image with horizontal orientation (width greater than height).
    *   `--portrait`: Generates an image with vertical orientation (height greater than width).
*   **Note:** These parameters are mutually exclusive.

### `--small` / `--medium` / `--large`

*   **Description:** Controls the size of the image that will be generated.
*   **Usage:**
    *   `--small`: Create a smaller image, which is faster to generate and uses less memory.
    *   `--medium`: Create an image of the size the model was trained for.
    *   `--large`: Generates a larger image with more detail. (recommended)
*   **Note:** These parameters are mutually exclusive.

### `--style <style>`

*   **Description:** Defines the general artistic style of the generated image.
*   **Example:** `--style PIXELART`
*   **Usage:** Allows choosing a predefined style for the image.
<!--
*   **Options:**
    *   `PHOTO`: Realistic photographic style.
    *   `DARKFAN80`: Dark fantasy style from the 80s.
    *   `PIXELART`: Pixel art style.
    *   `INK`: Ink drawing style.
    *   `CUTETOY`: Cute toy style.
    *   `1GIRLX`: ...
    *   `1GIRLZ`: ...
    *   `CUSTOM1`: Custom style 1.
    *   `CUSTOM2`: Custom style 2.
-->

### `--batch-size <number>`

*   **Description:** Specifies the number of images to generate in a batch.
*   **Example:** `--batch-size 4`
*   **Usage:** Allows generating multiple images at once.
*   **Values:** An integer indicating the batch size.


