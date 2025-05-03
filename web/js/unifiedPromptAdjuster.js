/**
 * File    : unifiedPromptAdjuster.js
 * Purpose : Allows to use the Ctrl+LEFT/RIGHT/UP/DOWN keys to adjust parameter values
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Jan 26, 2025
 * Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                              ComfyUI-TinyBreaker
 * ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
 *  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 The comfy js extension documentation is located at:
  - https://docs.comfy.org/custom-nodes/javascript_overview

 The original native ComfyUI extension for adjusting prompts is located at:
  - https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/extensions/core/editAttention.ts

 Since the native ComfyUI extension for adjusting prompts cannot be disabled
 and `unifiedPromptAdjuster` uses the same keys (CTRL+UP/DOWN) as ComfyUI,
 the code implemented here employs some 'creative' workarounds to address this issue.

 2025.01.31 The code was tested in Chrome version 132.0.6834.110
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
import { app } from "../../../scripts/app.js"
const ENABLED = true;

/**
 * List of aspect ratios that can be used as value for `--ratio` parameter.
 */
const ASPECT_RATIOS = [
    "1:1", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9", "12:5", "70:27", "32:9" ];
const DEFAULT_ASPECT_RATIO = "4:3"

/**
 * List of upscale values that can be used as value for `--upscale` parameter.
 */
const UPSCALE_VALUES = [
    "on", "off" ];
const DEFAULT_UPSCALE_VALUE = "off";

/**
 * List of upscale noise levels that can be used as value for `--upscale-noise` parameter.
 */
const UPSCALE_NOISE_LEVELS = [
    "none", "minimal", "low", "normal", "high", "veryhigh", "maximum" ];

/**
 * List of detail levels that can be used as value for `--detail-level` parameter.
 */
const DETAIL_LEVELS = [
    "none", "minimal", "low", "normal", "high", "veryhigh", "maximum" ];

/**
 * List of image sizes that can be used as parameters within a prompt.
 */
const IMAGE_SIZES = [
    "--medium", "--large" ];

/**
 * List of image orientations that can be used as parameters within a prompt.
 */
const IMAGE_ORIENTATIONS = [
    "--portrait", "--landscape" ];

/**
 * List of all options that the user can cycle through by pressing CTRL+LEFT/RIGHT
 */
const NAVIGABLE_OPTIONS = [
    "--no", "--refine", "--seed",
    "--cfg-shift", "--image-shift",
    "--aspect", "--landscape", "--portrait", "--medium",
    "--upscale", "--upscale-noise",
    "--batch-size", "--detail-level"
];

/**
 * List of options that can be automatically completed when pressing CTRL+RIGHT
 */
const AUTOCOMPLETE_LIST = [
    "--no", "--refine", "--seed",
    "--cfg-shift", "--image-shift",
    "--aspect", "--landscape", "--portrait", "--medium",
    "--upscale", "--upscale-noise",
    "--batch-size", "--detail-level"
];

/*------------------------------- HELPERS -------------------------------*/

/**
 * Finds the first widget with tagName.
 * @param {ComfyNode} node   : The ComfyUI node to search in.
 * @param {String}    tagName: The tag name to search for. e.g. "TEXTAREA" or "INPUT".
 * @returns 
 *     The first widget with the given tag name, or `null` if not found.
 */
function findWidgetInComfyNode(node, tagName) {
	if ( node?.widgets?.length ) {
		for (const widget of node.widgets) {
			if (widget.inputEl?.tagName === tagName) {
				return widget;
			}
		}
	}
    return null;
}

/**
 * Finds the first substring that matches the given regular expression in the given string
 * @param {String} str             : The string to search in.
 * @param {RegExp} regex           : The regular expression to search for. e.g. /--detail\s*([^ ]+)/
 * @param {Number} initialPosition : The position to start searching from.
 * @param {Number} defaultResult   : The value to return if no match was found.
 * @returns
 *   The index of the first substring that matches the given regular expression, or -1 if no match was found.
 */
function searchSubstring(str, regex, initialPosition = 0, defaultResult = -1) {
    if (initialPosition < 0) { initialPosition = 0; }
    const result = str.slice(initialPosition).search(regex);
    if (result < 0) { return defaultResult; }
    return result + initialPosition;
}

/**
 * Removes emphasis weights from a string.
 * ComfyUI forcefully intercepts CTRL+UP/DOWN and adds emphasis/weights,
 * this function is responsible for removing this extra emphasis from text,
 *
 * Example:
 *     "(--seed 400:1.2)" -> "--seed 400"
 *
 * @param {String} text: The string from which attention weights will be removed.
 * @returns
 *     The string without attention weights.
 */
function removeEmphasis(str) {
    const regex = /\(([^:]+):[^\)]+\)/;
    const match = str.match(regex);
    if (match && match[1]) {
        return match[1]
    }
    return str;
}

/**
 * Inserts text into a textarea at the specified range.
 * @param {HTMLTextAreaElement} textarea: The textarea element where the text will be inserted.
 * @param {Number} start: The start position of the range.
 * @param {Number} end  : The end position of the range.
 * @param {String} text : A string with the text to be inserted.
 */
function insertText(textarea, text, start, end) {
    // ATTENTION: intentional use of the execCommand() method to modify textarea content
    // (the method is deprecated but still works in modern browsers)
    // https://developer.mozilla.org/docs/Web/API/Document/execCommand#using_inserttext
    if (start !== undefined && end !== undefined) {
        textarea.setSelectionRange(start, end)
    }
    document.execCommand('insertText', false, text)
}

/*-------------------------- ADJUSTMENT PROCESS ---------------------------*/

/**
 * Adjusts the value of an integer argument.
 * @param {string}   name          - The name of the argument to be adjusted.
 * @param {string}   value         - The current value of the argument (string representation).
 * @param {number}   offset        - Value to add to the argument's current value.
 * @param {number}   defaultValue  - Default value to use if the input is empty or invalid.
 * @param {Object}   options                - Optional configuration parameters.
 * @param {number}   [options.min]          - Minimum allowed value for the adjusted number.
 * @param {number}   [options.max]          - Maximum allowed value for the adjusted number.
 * @param {string}   [options.positiveChar] - Character to prepend to positive numbers (e.g., '+').
 * @param {string[]} [options.textValues]   - Array of text values to use for non-numerical inputs.
 * @returns {string}
 *   A string representing the adjusted argument, in the format: "name [adjustedValue]"
 *   - The adjusted value is clamped to `min`/`max` (if provided) and formatted with optional `positiveChar`
 *   - A trailing space is included after the value (e.g., "argname -4 ").
 */
function adjustInt(name, value, offset, defaultValue, { min, max, positiveChar, textValues }) {
    value = value.trim();
    let   number    = 0;
    const oldNumber = parseInt(value);

    if( value === "" ) {
        // empty value is replaced with default value
        number = defaultValue ?? 0;
    } else if( isNaN(oldNumber) ) {
        // non-numerical value can be replaced with multiple text values if provided
        number = defaultValue ?? 0;
        if( textValues !== undefined ) { return adjustMultipleChoice(name, value, offset, defaultValue, {choices:textValues}); }
    } else {
        // finally, numerical values are adjusted by offset
        number = oldNumber + offset;
        if( min !== undefined && number < min ) { number = min; }
        if( max !== undefined && number > max ) { number = max; }
    }
    // convert number to string, and add positive char if necessary
    const intNumber = Math.trunc(number);
    let   strNumber = intNumber.toString();
    if (positiveChar && intNumber > 0) {
        strNumber = `${positiveChar}${strNumber}`;
    }
    // return the full argument string (the name and the adjusted number)
    return `${name} ${strNumber} `;
}

/**
 * Adjusts the value of a float argument.
 * @param {string}   name          - The name of the argument to be adjusted.
 * @param {string}   value         - The current value of the argument (string representation).
 * @param {number}   offset        - Value to add to the argument's current value.
 * @param {number}   defaultValue  - Default value to use if the input is empty or invalid.
 * @param {Object}   options                - Optional configuration parameters.
 * @param {number}   [options.min]          - Minimum allowed value for the adjusted number.
 * @param {number}   [options.max]          - Maximum allowed value for the adjusted number.
 * @param {string}   [options.positiveChar] - Character to prepend to positive numbers (e.g., '+').
 * @param {string[]} [options.textValues]   - Array of text values to use for non-numerical inputs.
 * @returns {string}
 *   A string representing the adjusted argument, in the format: "name [adjustedValue]"
 *   - The adjusted value is clamped to `min`/`max` (if provided) and formatted with optional `positiveChar`
 *   - A trailing space is included after the value (e.g., "argname +3.2 ").
 */
function adjustFloat(name, value, offset, defaultValue, { min, max, positiveChar, textValues }) {
    value = value.trim();
    let   number    = 0.0;
    const oldNumber = parseFloat(value);

    if( value === "" ) {
        // empty value is replaced with default value
        number = defaultValue ?? 0.0;
    } else if( isNaN(oldNumber) ) {
        // non-numerical value can be replaced with multiple text values if provided
        number = defaultValue ?? 0.0;
        if( textValues !== undefined ) { return adjustMultipleChoice(name, value, offset, defaultValue, {choices:textValues}); }
    } else {
        // finally, numerical values are adjusted by offset
        number = oldNumber + offset;
        if( min !== undefined && number < min ) { number = min; }
        if( max !== undefined && number > max ) { number = max; }
    }
    // convert number to string, and add positive char if necessary
    let strNumber = number.toFixed(1);
    if (positiveChar && number >= 0.001) {
        strNumber = `${positiveChar}${strNumber}`;
    }
    // return the full argument string (the name and the adjusted number)
    return `${name} ${strNumber} `;
}

/**
 * Adjusts the value of a multiple choice argument.
 * @param {string}   name         - The name of the argument to be adjusted.
 * @param {string}   value        - The current value of the argument (string representation).
 * @param {number}   offset       - Offset indicating the direction (positive = next value, negative = previous value).
 * @param {string}   defaultValue - Default value to use if the input is empty or invalid.
 * @param {Object}   options              - Optional configuration parameters.
 * @param {string[]} [options.choices]    - Array of valid choices for the multiple choice argument.
 * @param {boolean}  [options.isCircular] - Flag indicating if the selection should wrap around (circular mode). Defaults to false.
 * @returns {string}
 *     The formatted argument string with the adjusted multiple choice value.
 */
function adjustMultipleChoice(name, value, offset, defaultValue, { choices, isCircular }) {
    value   = value.trim()
    choices = choices ?? ["True", "False"];
    let   index     = 0;
    const oldIndex  = choices.indexOf( value );
    if( oldIndex>=0 ) {
        if     ( offset>0 ) { index = oldIndex + 1; }
        else if( offset<0 ) { index = oldIndex - 1; }
    }
    else if( defaultValue ) {
        index = Math.max( 0, choices.indexOf(defaultValue) )
    }
    // adjusts index within limits or wrap-around if circular mode
    if( isCircular ) { index = (index + choices.length) % choices.length;          }
    else             { index = Math.min( Math.max( 0, index ), choices.length -1); }
    return name ? `${name} ${choices[index]} ` : `${choices[index]} `;
}

/**
 * The main function that adjusts the value of an argument based on its type.
 * @param {String} name  : The name of the argument to be adjusted.
 * @param {String} value : The current value of the argument.
 * @param {Number} offset: The amount by which the argument's value will be adjusted.
 * @returns
 *   The full argument string with the adjusted value (including the argument's name)
 */
function adjustArgument(name, value, offset) {
    switch(name) {
        case '--seed':
            return adjustInt(name, value, offset, 1, {min:1});
        case '--cfg-shift':
            return adjustInt(name, value, offset, 0, {min:-20, max:20, positiveChar:'+'});
        case '--image-shift':
            return adjustInt(name, value, offset, 0, {min:0});
        case '--upscale':
            return adjustMultipleChoice(name, value, offset, DEFAULT_UPSCALE_VALUE, {choices:UPSCALE_VALUES, isCircular:true});
        case '--upscale-noise':
            return adjustInt(name, value, offset, 0, {min:-5, max:5, positiveChar:'+', textValues: UPSCALE_NOISE_LEVELS});
        case '--detail-level':
            return adjustInt(name, value, offset, 0, {min:-9, max:9, positiveChar:'+', textValues: DETAIL_LEVELS});
        case '--aspect':
            return adjustMultipleChoice(name, value, offset, DEFAULT_ASPECT_RATIO, {choices:ASPECT_RATIOS, isCircular:true});
        case '--batch-size':
            return adjustInt(name, value, offset, 1, {min:1});
    }
    if( IMAGE_ORIENTATIONS.includes(name) ) {
        return adjustMultipleChoice("", name, offset, "", {choices:IMAGE_ORIENTATIONS, isCircular:true});
    }
    if( IMAGE_SIZES.includes(name) ) {
        return adjustMultipleChoice("", name, offset, "", {choices:IMAGE_SIZES});
    }
    return null;
}

/**
 * Returns the string that is closest to `partialString` in a list of available strings.
 * @param {String}        partialString          : The string that the user has typed so far.
 * @param {Array<String>} listOfAvailableStrings : An array with all possible strings available to choose from.
 * @returns
 *   The string that is closest to `partialString`, or `null` if no such string was found.
 */
function Autocomplete(partialString, listOfAvailableStrings) {
    if( !partialString ) { return null; }
    const partialLowercase = partialString.toLowerCase();
    for( const available of listOfAvailableStrings ) {
        if( available.startsWith(partialLowercase) && available !== partialLowercase ) {
            return available
        }
    }
    return null;
}

/*------------------------------ KEY EVENTS -------------------------------*/

/**
 * Function to handle left/right arrow key presses.
 * @param {Boolean}        isMovingRight: Indicates whether the key press was a right arrow.
 * @param {KeyboardEvent}       event   : The keyboard event object.
 * @param {HTMLTextAreaElement} textarea: The textarea element where text will be modified.
 */
function onLeftOrRight(isMovingRight, event, textarea) {

    let selectionStart = textarea.selectionStart;
    let selectionEnd   = textarea.selectionEnd;
    const text         = textarea.value;
    const before       = text.substring(selectionStart-3, selectionStart).padStart(3,' ');
    const after        = text.substring(selectionEnd, selectionEnd+3).padEnd(3,' ');
    const selectedText = text.substring(selectionStart, selectionEnd);

    let new_option     = null;
    let cursorPosition = -1;

    //-- AUTOCOMPLETE THE OPTION --------------------------

    // if CTRL+RIGHT and no text selected and not after a '--',
    // try to autocomplete the argument's name behind the cursor
    if( isMovingRight && selectionStart==selectionEnd && before.charAt(2) != '-' ) {
        selectionStart    = text.lastIndexOf("--", selectionStart + 2);
        const argumentEnd = selectionStart>=0 ? searchSubstring(text, /\s/, selectionStart, text.length) : -1;
        if( selectionEnd === argumentEnd ) {
            const partion_option = text.substring(selectionStart, selectionEnd);
            new_option = Autocomplete(partion_option, AUTOCOMPLETE_LIST)
            if( new_option ) {
                cursorPosition = selectionStart + new_option.length;
            }
        }
    }

    //-- SELECTION THROUGH `NAVIGABLE_OPTIONS` ------------

    // if nothing is selected and the cursor if between '--' and a space
    // then the new option will be the first/last option from the list
    if( selectionStart==selectionEnd &&
        before.charAt(0).trim()==='' && before.charAt(1)==='-' && before.charAt(2)==='-' &&
        after.charAt(0).trim()==='' )
    {
        let index
        if ( isMovingRight ) { index = 0; }
        else                 { index = NAVIGABLE_OPTIONS.length - 1; }
        new_option = NAVIGABLE_OPTIONS[index % NAVIGABLE_OPTIONS.length];
        selectionStart -= 2
    }
    // if the selected text is '--'
    // then the new option will be the next/previous option from the list
    else if( selectedText==="--" )
    {
        selectionEnd = searchSubstring(text, /\s/, selectionEnd, text.length);
        const option = text.substring(selectionStart, selectionEnd)
        let index = NAVIGABLE_OPTIONS.indexOf(option)
        if( index<0 ) { return }

        if( isMovingRight ) { index += 1; }
        else                { index += NAVIGABLE_OPTIONS.length - 1; }
        new_option = NAVIGABLE_OPTIONS[index % NAVIGABLE_OPTIONS.length];
    }

    //-- FINAL SUBSTITUTION -------------------------------

    // if a new option was selected by the above logic,
    // insert it into the textarea
    if( new_option ) {
        insertText(textarea, new_option, selectionStart, selectionEnd);
        if( cursorPosition >= 0 ) { textarea.setSelectionRange(cursorPosition, cursorPosition);   }
        else                      { textarea.setSelectionRange(selectionStart, selectionStart+2); }
        event.preventDefault();
    }
}

/**
 * Function to handle up/down arrow key presses.
 * @param {Boolean}         isMovingDown: Indicates whether the key press was a down arrow.
 * @param {KeyboardEvent}       event   : The keyboard event object.
 * @param {HTMLTextAreaElement} textarea: The textarea element where text will be modified.
 */
function onUpOrDown(isMovingDown, event, textarea) {
    event.preventDefault()

    // extracts the argument quickly and provisionally (only 8 characters)
    const selectionStart = textarea.selectionStart;
    const selectionEnd   = textarea.selectionEnd;
    let   text           = textarea.value;
    let   argumentStart  = text.lastIndexOf("--", selectionStart + 2);
    let   argument       = text.substring(argumentStart, argumentStart+9)

    // if the argument that is being modified is some of the "prompt" arguments
    // (prompt, negative prompt, refine prompt) then we should keep the modification that ComfyUI already made.
    if( !argument.startsWith("--") || argument.startsWith("--no ") || argument.startsWith("--refine ") ) {
        return;
    }

    // remove any emphasis weights added by the native ComfyUI extension
    let selectedText        = text.substring(selectionStart, selectionEnd);
    let textWithoutEmphasis = removeEmphasis(selectedText);
    if( textWithoutEmphasis !== selectedText ) {
        insertText(textarea, textWithoutEmphasis, selectionStart, selectionEnd);
        text           = textarea.value;
        argumentStart  = text.lastIndexOf("--", selectionStart + 2);
    }

    // re-extract the complete argument,
    // (this time correctly since there is no emphasis in the selected text)
    const argumentEnd = Math.min(
        searchSubstring(text, "--"        , argumentStart+2, text.length),
        searchSubstring(text, /[\r\n\f\v]/, argumentStart+2, text.length)
    );
    argument = text.substring(argumentStart, argumentEnd);

    // split the argument into name and value
    let nameLength = argument.indexOf(" ");
    if( nameLength === -1 ) { nameLength = argument.length; }
    let argumentName  = argument.substring(0, nameLength);
    let argumentValue = argument.substring(nameLength);

    // adjust the value of the argument based on the key pressed
    argument = adjustArgument(argumentName, argumentValue, isMovingDown ? -1 : 1);
    if (argument !== null) {
        insertText(textarea, argument, argumentStart, argumentEnd);
        nameLength = argument.indexOf(" ");
        if( nameLength === -1 ) { nameLength = argument.length; }
    }

    // select the name of the argument
    textarea.setSelectionRange(argumentStart, argumentStart+nameLength);
}

/**
 * Function to be called every time a key is pressed.
 * @param {KeyboardEvent} event: The keyboard event object.
 */
function onKeyDown(event) {
    /** @type {HTMLTextAreaElement} */
    const textarea = event.composedPath()[0];

    // check if Ctrl or Cmd is being held down
    if( !event.ctrlKey && !event.metaKey ) { return; }

    // check if the target element is a textarea for unified prompts
    if( textarea.tagName !== 'TEXTAREA' ) { return; }
    if( !textarea.isUnifiedPrompt       ) { return; }

    if( event.key === 'ArrowUp' || event.key === 'ArrowDown' ) {
        onUpOrDown(event.key === 'ArrowDown', event, textarea)
    }
    else if( event.key === 'ArrowLeft' || event.key === 'ArrowRight' ) {
        onLeftOrRight(event.key === 'ArrowRight', event, textarea);
    }
    return
}


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({
    name: "TinyBreaker.unifiedPromptAjuster",

    /**
     * Called when the extension is loaded.
     */
    init() {
        if (!ENABLED) return;
        console.log("##>> Unified Prompt Adjuster extension loaded.")
        window.addEventListener('keydown', onKeyDown)
    },

	/**
	 * Called every time ComfyUI creates a new node.
	 * @param {ComfyNode} node - The node that was created.
	 */
	async nodeCreated(node) {
		if (!ENABLED) return;

        // only applies to "Unified Prompt" nodes with a textarea
		if( !node?.comfyClass?.startsWith("UnifiedPromptInput")  ) { return; }
        const widget = findWidgetInComfyNode(node, 'TEXTAREA');
        if( !widget?.inputEl ) { return; }

        // mark the textarea as a unified prompt
        widget.inputEl.isUnifiedPrompt = true;
	},


})

