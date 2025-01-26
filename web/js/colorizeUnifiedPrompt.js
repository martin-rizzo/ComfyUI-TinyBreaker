/**
 * File    : colorizeUnifiedPrompt.js
 * Purpose : Colorizes the text entered by the user in the unified prompt node.
 * Author  : Martin Rizzo | <martinrizzo@gmail.com>
 * Date    : Jan 25, 2025
 * Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
 * License : MIT
 *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 *                              ConfyUI-TinyBreaker
 * ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
 *  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
 *_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
*/
import { app } from "../../../scripts/app.js"
const ENABLED = false;

// JS Extensions Documentation:
//   https://docs.comfy.org/custom-nodes/javascript_overview
//
// called when a node is created
// // node
//  + widgets[]
//      + inputEl/element
//      + options
//



class UnifiedPromptHighlighter {
    constructor(textarea, options = {}) {
        this.textarea = textarea;
        this.options = {
            highlightClass: "comfy-multiline-input", // 'highlighted-text', 
            ...options,
        };
        this.highlightElement = this.createHighlightElement();
        this.textarea.parentNode.insertBefore(this.highlightElement, this.textarea.nextSibling);
        this.textarea.addEventListener('input', this.onInput);
        this.textarea.addEventListener('scroll', this.onScroll);

		// initialize the highlighter after a short delay
		// to allow the textarea content to be updated
		this.timerId        = setInterval(this.onDelayedInitialization, 250);
		this.timerCountDown = 10;

		// observe changes to the textarea's style attribute
		// to detect changes to the textarea's position and scale
        this.observer = new MutationObserver(this.onStyleChange);
        this.observer.observe(this.textarea, { attributes: true, attributeFilter: ['style'] });
		this.onStyleChange();
    }

	onDelayedInitialization = () => {
		console.log("##>> delayed initialization!!");
		this.timerCountDown -= 1;
		if ( this.textarea.value != "" || this.timerCountDown <= 0 ) {
			clearInterval(this.timerId);
			this.onInput();
		}
    }

    onStyleChange = () => {
        const element = this.highlightElement;
        const styleObj = this.textarea.style;
        for (let i = 0; i < styleObj.length; i++) {
            const property = styleObj[i];
            if (property == "will-change" || property == "display") {
                continue;
            }
            const value = styleObj.getPropertyValue(property);
            element.style.setProperty(property, value);
        }
        element.style.setProperty("pointer-events", "none");
    }

    onInput = () => {
        const text = this.textarea.value;
        const highlightedText = this.applyHighlighting(text);
        this.highlightElement.innerHTML = highlightedText;
    };

    onScroll = () => {
        this.highlightElement.scrollTop = this.textarea.scrollTop;
        this.highlightElement.scrollLeft = this.textarea.scrollLeft;
    };

    createHighlightElement() {
        const element = document.createElement('pre');
        element.classList.add(this.options.highlightClass);
        return element;
    }


    applyHighlighting(text) {
        // Lógica de resaltado aquí (ejemplo básico)
        const highlighted = text.replace(/(\w+)/g, '<span style="color: blue;">$1</span>');
        return highlighted;
    }

    remove() {
		clearInterval(this.timerId)
        this.observer.disconnect();
        this.textarea.removeEventListener('input', this.onInput);
        this.textarea.removeEventListener('scroll', this.onScroll);
        this.highlightElement.remove();
        this.textarea         = null;
        this.highlightElement = null;
    }
}


/**
 * Find the textarea widget in a given ComfyUI node.
 * @param {ComfyNode} node - The ComfyUI node where to search for the textarea.
 * @returns 
 *   The textarea widget or null if not found.
 */
function findTextareaWidget(node) {
	if ( node?.widgets?.length ) {
		for (const widget of node.widgets) {
			if (widget.inputEl?.tagName === 'TEXTAREA') {
				return widget;
			}
		}
	}
    return null;
}


//#=========================================================================#
//#////////////////////////// REGISTER EXTENSION ///////////////////////////#
//#=========================================================================#

app.registerExtension({

	name: "TinyBreaker.colorizeUnifiedPrompt",

	/**
	 * Called every time ComfyUI creates a new node.
	 * @param {ConfyNode} node - The node that was created.
	 */
	async nodeCreated(node) {
		if (!ENABLED) return;
        // console.log("##>> New node created: ", node.title)

		// only applies to "Unified Prompt" nodes with a textarea
		if ( !node?.title.includes("Unified Prompt") ) return;
		const widget = findTextareaWidget(node);
		if (!widget || !widget.inputEl) return;

		// create the highlight element and add it to the DOM
		const unifiedPromptHighlighter = new UnifiedPromptHighlighter(widget.inputEl);
		const onRemoved = node.onRemoved
		node.onRemoved = function () {
			unifiedPromptHighlighter.remove();
			onRemoved?.apply(this, arguments)
		}
	},

})

