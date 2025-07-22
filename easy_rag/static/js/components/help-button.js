/**
 * Help Button Component
 * 
 * This component creates a help button that opens a help modal when clicked.
 */
class HelpButton extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        const helpId = this.getAttribute('help-id') || '';
        const buttonText = this.getAttribute('text') || 'Help';
        const buttonSize = this.getAttribute('size') || '';
        const buttonVariant = this.getAttribute('variant') || 'link';
        
        // Determine button class based on variant and size
        let buttonClass = `btn btn-${buttonVariant}`;
        if (buttonSize) {
            buttonClass += ` btn-${buttonSize}`;
        }
        
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: inline-block;
                }
            </style>
            <button type="button" class="${buttonClass}" data-help-id="${helpId}">
                <i class="bi bi-info-circle me-1"></i>${buttonText}
            </button>
        `;
        
        // Add event listener
        const button = this.shadowRoot.querySelector('button');
        button.addEventListener('click', (e) => {
            e.preventDefault();
            this.showHelpModal(helpId);
        });
    }
    
    /**
     * Show help modal
     */
    showHelpModal(helpId) {
        fetch(`/help/${helpId}`)
            .then(response => response.json())
            .then(data => {
                // Create modal
                const modalHtml = `
                    <div class="modal fade" id="helpModal" tabindex="-1" aria-labelledby="helpModalLabel" aria-hidden="true">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title" id="helpModalLabel">${data.title}</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div class="modal-body">
                                    ${data.content}
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add modal to body
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('helpModal'));
                modal.show();
                
                // Remove modal from DOM when hidden
                document.getElementById('helpModal').addEventListener('hidden.bs.modal', function() {
                    this.remove();
                });
            })
            .catch(error => {
                console.error('Error fetching help content:', error);
                alert('Failed to load help content. Please try again later.');
            });
    }
}

// Define the custom element
customElements.define('help-button', HelpButton);