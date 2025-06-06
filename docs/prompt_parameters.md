# Prompt Parameters

One of the advantages of the **Unified Prompt Input** is that it allows parameters to be appended after the prompt, facilitating modifications in image generation. These parameters always start with two hyphens (`--`), offering an easy way to adjust the final output.

- Parameters for Minor Adjustments
  - [`--no`](#--no-elements)
  - [`--refine`](#--refine-description)
  - [`--cfg-shift`](#--cfg-shift-number)
  - [`--image-shift`](#--image-shift-number)
  - ['--upscale`](#-upscale)]
- Parameters for Major Changes
  - [`--seed`](#--seed-number)
  - [`--aspect`](#--aspect-ratio)
  - [`--landscape` / `--portrait`](#--landscape----portrait)
  - [`--medium`](#--small----medium----large)
- Extra parameters
  - [`--detail-level`](#--detail-level-level)
  - [`--batch-size`](#--batch-size-number)

## Minor Adjustments

These parameters allow for subtle modifications and fine-tuning of the generated image.

### `--no <text>`

*   **Description:** Specifies elements that should **not** appear in the generated image. (negative prompt)
*   **Example:** `--no trees, sky, clouds`
*   **Usage:** Useful for removing unwanted elements from the final result.

### `--refine <text>`

*   **Description:** Provides a textual description of what elements should be refined in the image.
*   **Example:** `--refine trees and cat
*   **Usage:** Helps to refine specific elements in the image.

### `--image-shift <number>`

*   **Description:** Specifies variants of the refinement process without changing the composition of the image.
*   **Example:** `--image-shift 2`
*   **Usage:** Although the image remains practically unchanged, different variants can solve errors in hands and faces.
*   **Values:** An integer indicating the desired variant.

### `--cfg-shift <number>`

*   **Description:** Adjusts the correction value to the default Classifier-Free Guidance (CFG).
*   **Example:** `--cfg-shift -2`
*   **Usage:** Allows fine-tuning the effect of the Classifier-Free Guidance (CFG).
*   **Values:** A floating-point number. A value of `0.0` is the best value for the current style.

### `--detail-level <level>`

*   **Description:** The intensity of the refiner.
*   **Example:** `--detail-level normal`
*   **Usage:** Adjusts the level of detail added during the refinement process.
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
    *   `--large`: Generates a larger image with more detail. (default)
    *   `--medium`: Create an image of the size the model was trained for.
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


