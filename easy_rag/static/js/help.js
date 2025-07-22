/**
 * Help system for Easy RAG System
 * 
 * This file contains functionality for tooltips, guided tours, and help popups
 */

// Initialize the help system
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    setupHelpButtons();
    setupGuidedTours();
    enhanceErrorMessages();
});

/**
 * Initialize tooltips on elements with data-help attribute
 */
function initializeTooltips() {
    const helpElements = document.querySelectorAll('[data-help]');
    helpElements.forEach(element => {
        // Create tooltip element
        const helpIcon = document.createElement('i');
        helpIcon.className = 'bi bi-question-circle text-primary ms-1';
        helpIcon.setAttribute('data-bs-toggle', 'tooltip');
        helpIcon.setAttribute('data-bs-placement', 'top');
        helpIcon.setAttribute('title', element.getAttribute('data-help'));
        
        // Add tooltip after the element
        element.insertAdjacentElement('afterend', helpIcon);
    });
    
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Setup help buttons that show detailed help modals
 */
function setupHelpButtons() {
    const helpButtons = document.querySelectorAll('.help-button');
    helpButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const helpId = this.getAttribute('data-help-id');
            showHelpModal(helpId);
        });
    });
}

/**
 * Show a help modal with the specified ID
 */
function showHelpModal(helpId) {
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
        });
}

/**
 * Setup guided tours for different workflows
 */
function setupGuidedTours() {
    const tourButtons = document.querySelectorAll('.tour-button');
    tourButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const tourId = this.getAttribute('data-tour-id');
            startGuidedTour(tourId);
        });
    });
}

/**
 * Start a guided tour with the specified ID
 */
function startGuidedTour(tourId) {
    fetch(`/help/tour/${tourId}`)
        .then(response => response.json())
        .then(tourData => {
            // Check if Shepherd.js is loaded
            if (typeof Shepherd === 'undefined') {
                // Load Shepherd.js dynamically
                const shepherdCss = document.createElement('link');
                shepherdCss.rel = 'stylesheet';
                shepherdCss.href = 'https://cdn.jsdelivr.net/npm/shepherd.js@10.0.1/dist/css/shepherd.css';
                document.head.appendChild(shepherdCss);
                
                const shepherdJs = document.createElement('script');
                shepherdJs.src = 'https://cdn.jsdelivr.net/npm/shepherd.js@10.0.1/dist/js/shepherd.min.js';
                document.head.appendChild(shepherdJs);
                
                shepherdJs.onload = function() {
                    initializeTour(tourData);
                };
            } else {
                initializeTour(tourData);
            }
        })
        .catch(error => {
            console.error('Error fetching tour data:', error);
        });
}

/**
 * Initialize a tour with the provided tour data
 */
function initializeTour(tourData) {
    const tour = new Shepherd.Tour({
        useModalOverlay: true,
        defaultStepOptions: {
            cancelIcon: {
                enabled: true
            },
            classes: 'shadow-md bg-purple-dark',
            scrollTo: true
        }
    });
    
    tourData.steps.forEach(step => {
        tour.addStep({
            id: step.id,
            title: step.title,
            text: step.content,
            attachTo: {
                element: step.element,
                on: step.position || 'bottom'
            },
            buttons: [
                {
                    text: 'Back',
                    action: tour.back,
                    classes: 'btn btn-secondary',
                    disabled: step.isFirst
                },
                {
                    text: step.isLast ? 'Finish' : 'Next',
                    action: step.isLast ? tour.complete : tour.next,
                    classes: 'btn btn-primary'
                }
            ]
        });
    });
    
    tour.start();
}

/**
 * Enhance error messages with explanations
 */
function enhanceErrorMessages() {
    const errorMessages = document.querySelectorAll('.alert-danger, .error-message');
    errorMessages.forEach(message => {
        // Check if the message has a data-error-code attribute
        const errorCode = message.getAttribute('data-error-code');
        if (errorCode) {
            // Add a help button
            const helpButton = document.createElement('button');
            helpButton.className = 'btn btn-sm btn-link text-danger';
            helpButton.innerHTML = '<i class="bi bi-info-circle"></i> Learn more';
            helpButton.addEventListener('click', function(e) {
                e.preventDefault();
                showErrorExplanation(errorCode);
            });
            
            message.appendChild(helpButton);
        }
    });
}

/**
 * Show an explanation for an error code
 */
function showErrorExplanation(errorCode) {
    fetch(`/help/error/${errorCode}`)
        .then(response => response.json())
        .then(data => {
            // Create modal
            const modalHtml = `
                <div class="modal fade" id="errorModal" tabindex="-1" aria-labelledby="errorModalLabel" aria-hidden="true">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="errorModalLabel">Error: ${data.title}</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                <h6>What happened:</h6>
                                <p>${data.explanation}</p>
                                <h6>How to fix it:</h6>
                                <p>${data.solution}</p>
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
            const modal = new bootstrap.Modal(document.getElementById('errorModal'));
            modal.show();
            
            // Remove modal from DOM when hidden
            document.getElementById('errorModal').addEventListener('hidden.bs.modal', function() {
                this.remove();
            });
        })
        .catch(error => {
            console.error('Error fetching error explanation:', error);
        });
}