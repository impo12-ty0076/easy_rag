/**
 * Help Icon Component
 * 
 * This component creates a help icon with a tooltip that can be added to any element.
 */
class HelpIcon extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        const text = this.getAttribute('text') || 'No help text provided';
        const position = this.getAttribute('position') || 'top';
        
        this.shadowRoot.innerHTML = `
            <style>
                .help-icon {
                    display: inline-block;
                    color: #0d6efd;
                    cursor: pointer;
                    margin-left: 0.25rem;
                    font-size: 0.9rem;
                }
                .help-icon:hover {
                    color: #0a58ca;
                }
            </style>
            <i class="bi bi-question-circle help-icon" data-bs-toggle="tooltip" data-bs-placement="${position}" title="${text}"></i>
        `;
        
        // Initialize tooltip
        setTimeout(() => {
            const icon = this.shadowRoot.querySelector('.help-icon');
            new bootstrap.Tooltip(icon);
        }, 100);
    }
}

// Define the custom element
customElements.define('help-icon', HelpIcon);