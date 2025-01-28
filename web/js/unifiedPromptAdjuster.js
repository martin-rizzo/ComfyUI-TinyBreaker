/**
 * File    : unifiedPromptAdjuster.js
 * Purpose : Allows to use the Ctrl+UP/DOWN keys to adjust parameter values
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Jan 26, 2025
 * Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                              ConfyUI-TinyBreaker
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

 2024/01/27: The code was tested in Chrome version 132.0.6834.110
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
import { app } from "../../../scripts/app.js"
const ENABLED = true;

/**
 * List of aspect ratios that can be used as value for --ratio argument.
 */
const ASPECT_RATIOS = [
    "1:1",
    "4:3",
    "3:2",
    "16:10",
    "16:9",
    "2:1",
    "21:9",
    "12:5",
    "70:27",
    "32:9",
]

/**
 * List of image sizes that can be used as an argument.
 */
const IMAGE_SIZES = [
    "--small",
    "--medium",
    "--large",
]

/**
 * List of image orientations that can be used as an argument.
 */
const IMAGE_ORIENTATIONS = [
    "--portrait",
    "--landscape"
]

/**
 * List of image formats that can be used as an argument.
 * (formats are a combination of image size and orientation)
 */
const IMAGE_FORMATS = [
    "--small-portrait",
    "--small-landscape",
    "--medium-portrait",
    "--medium-landscape",
    "--large-portrait",
    "--large-landscape",
]

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
 * Removes emphasis weights from a string.
 * ConfyUI forcefully intercepts CTRL+UP/DOWN and adds emphasis/weights,
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
function insertText(textarea, start, end, text) {
    // ATTENTION: intentional use of the execCommand() method to modify textarea content
    // (the method is deprecated but still works in modern browsers)
    // https://developer.mozilla.org/docs/Web/API/Document/execCommand#using_inserttext
    textarea.setSelectionRange(start, end)
    document.execCommand('insertText', false, text)
}

/**
 * Adjusts the value of an integer argument.
 * @param {String} name  : The name of the argument to be adjusted.
 * @param {String} value : The current value of the argument.
 * @param {Number} offset: The amount by which the argument's value will be adjusted.
 * @returns
 *   The full argument string with the adjusted value (including the argument's name)
 */
function adjustInt(name, value, offset) {
    const new_value = parseInt(value) + offset;
    return `${name} ${new_value} `;
}

/**
 * Adjusts the value of a float argument.
 * @param {String} name  : The name of the argument to be adjusted.
 * @param {String} value : The current value of the argument.
 * @param {Number} offset: The amount by which the argument's value will be adjusted.
 * @returns
 *   The full argument string with the adjusted value (including the argument's name)
 */
function adjustFloat(name, value, offset) {
    const new_value = parseFloat(value) + offset;
    return `${name} ${new_value.toFixed(1)} `;
}

/**
 * Adjusts the value of an argument that can be one of a set of predefined values.
 * @param {String}        name   : The name of the argument to be adjusted.
 * @param {String}        value  : The current value of the argument.
 * @param {Number}        offset : The amount by which the argument's value will be adjusted.
 * @param {Array<String>} options: An array of possible values for the argument.
 * @returns
 *   The full argument string with the adjusted value (including the argument's name)
 */
function adjustOption(name, value, offset, options) {
    const index = options.indexOf( value.trim() );
    let new_index = index>=0 ? index : 0;
    if     ( offset>0 ) { new_index += 1; }
    else if( offset<0 ) { new_index += options.length - 1; }
    const new_value = options[new_index % options.length];
    return name ? `${name} ${new_value} ` : `${new_value} `;
}

/*-------------------------- ADJUSTMENT PROCESS ---------------------------*/

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
        case '--cfg':
            return adjustFloat(name, value, offset*0.1);
        case '--seed':
            return adjustInt(name, value, offset);
        case '--variant':
            return adjustInt(name, value, offset)
        case '--batch':
            return adjustInt(name, value, offset)
        case '--ratio':
            return adjustOption(name, value, offset, ASPECT_RATIOS)
    }
    if( IMAGE_FORMATS.includes(name) ) {
        return adjustOption("", name, offset, IMAGE_FORMATS)
    }
    return null;
}

/**
 * Function to be called every time a key is pressed.
 * @param {KeyboardEvent} event: The keyboard event object.
 */
function onKeyDown(event) {
    /** @type {HTMLTextAreaElement} */
    const textarea = event.composedPath()[0];

    // check if the pressed key is an arrow key and if it's being held down with Ctrl or Cmd
    if( !event.ctrlKey          && !event.metaKey            ) { return; }
    if( event.key !== 'ArrowUp' && event.key !== 'ArrowDown' ) { return; }

    // check if the target element is a textarea for unified prompts
    if( textarea.tagName !== 'TEXTAREA' ) { return; }
    if( !textarea.isUnifiedPrompt       ) { return; }

    event.preventDefault()

    // extracts the argument quickly and provisionally (only 8 characters)
    let selectionStart = textarea.selectionStart
    let selectionEnd   = textarea.selectionEnd
    let argumentStart  = textarea.value.lastIndexOf("--", selectionStart + 2);
    let argument = textarea.value.substring(argumentStart, argumentStart+8)

    // if the argument that is being modified does not start with "--",
    // this means it's the main prompt and we should keep the modification that ComfyUI already made
    if( !argument.startsWith("--") ) {
        return
    }
    // if the argument that is being modified starts with "--no ",
    // this means it's the negative prompt and we should keep the modification that ComfyUI already made
    if( argument.startsWith("--no ") ) {
        return
    }

    // remove any emphasis weights added by the native ComfyUI extension
    let selectedText        = textarea.value.substring(selectionStart, selectionEnd)
    let textWithoutEmphasis = removeEmphasis(selectedText)
    if( textWithoutEmphasis !== selectedText ) {
        insertText(textarea, selectionStart, selectionEnd, textWithoutEmphasis)
        argumentStart  = textarea.value.lastIndexOf("--", selectionStart + 2);
    }

    // re-extract the complete argument,
    // this time correctly since there is no emphasis in the selected text
    let argumentEnd = textarea.value.indexOf("--", argumentStart + 2)
    if( argumentEnd === -1) { argumentEnd = textarea.value.length; }
    argument = textarea.value.substring(argumentStart, argumentEnd);

    // split the argument into name and value
    let nameLength = argument.indexOf(" ");
    if( nameLength === -1 ) { nameLength = argument.length; }
    let argumentName  = argument.substring(0, nameLength);
    let argumentValue = argument.substring(nameLength);

    // adjust the value of the argument based on the key pressed
    argument = adjustArgument(argumentName, argumentValue, event.key === 'ArrowUp' ? 1 : -1);
    if (argument !== null) {
        insertText(textarea, argumentStart, argumentEnd, argument)
        nameLength = argument.indexOf(" ");
        if( nameLength === -1 ) { nameLength = argument.length; }
    }

    // select the name of the argument
    textarea.setSelectionRange(argumentStart, argumentStart+nameLength)
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
	 * @param {ConfyNode} node - The node that was created.
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

