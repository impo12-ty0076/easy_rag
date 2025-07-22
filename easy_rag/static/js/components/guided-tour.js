/**
 * Guided Tour Component
 * 
 * This component creates a guided tour that walks users through key workflows.
 */
class GuidedTour {
    constructor(tourId) {
        this.tourId = tourId;
        this.steps = [];
        this.currentStep = 0;
        this.tour = null;
    }

    /**
     * Load tour data from the server
     */
    async loadTourData() {
        try {
            const response = await fetch(`/help/tour/${this.tourId}`);
            if (!response.ok) {
                throw new Error(`Failed to load tour data: ${response.status}`);
            }
            const tourData = await response.json();
            this.steps = tourData.steps;
            this.title = tourData.title;
            return tourData;
        } catch (error) {
            console.error('Error loading tour data:', error);
            return null;
        }
    }

    /**
     * Load Shepherd.js if not already loaded
     */
    async loadShepherd() {
        return new Promise((resolve, reject) => {
            if (typeof Shepherd !== 'undefined') {
                resolve();
                return;
            }

            // Load CSS
            const shepherdCss = document.createElement('link');
            shepherdCss.rel = 'stylesheet';
            shepherdCss.href = 'https://cdn.jsdelivr.net/npm/shepherd.js@10.0.1/dist/css/shepherd.css';
            document.head.appendChild(shepherdCss);
            
            // Load JS
            const shepherdJs = document.createElement('script');
            shepherdJs.src = 'https://cdn.jsdelivr.net/npm/shepherd.js@10.0.1/dist/js/shepherd.min.js';
            shepherdJs.onload = () => resolve();
            shepherdJs.onerror = () => reject(new Error('Failed to load Shepherd.js'));
            document.head.appendChild(shepherdJs);
        });
    }

    /**
     * Initialize the tour
     */
    async initialize() {
        try {
            // Load tour data and Shepherd.js in parallel
            const [tourData] = await Promise.all([
                this.loadTourData(),
                this.loadShepherd()
            ]);

            if (!tourData) {
                throw new Error('Failed to load tour data');
            }

            // Create tour
            this.tour = new Shepherd.Tour({
                useModalOverlay: true,
                defaultStepOptions: {
                    cancelIcon: {
                        enabled: true
                    },
                    classes: 'shadow-md bg-purple-dark',
                    scrollTo: true
                }
            });

            // Add steps
            this.steps.forEach(step => {
                this.tour.addStep({
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
                            action: this.tour.back,
                            classes: 'btn btn-secondary',
                            disabled: step.isFirst
                        },
                        {
                            text: step.isLast ? 'Finish' : 'Next',
                            action: step.isLast ? this.tour.complete : this.tour.next,
                            classes: 'btn btn-primary'
                        }
                    ]
                });
            });

            return true;
        } catch (error) {
            console.error('Error initializing tour:', error);
            return false;
        }
    }

    /**
     * Start the tour
     */
    async start() {
        if (!this.tour) {
            const initialized = await this.initialize();
            if (!initialized) {
                console.error('Failed to initialize tour');
                return false;
            }
        }

        this.tour.start();
        return true;
    }
}

// Register tour buttons
document.addEventListener('DOMContentLoaded', function() {
    const tourButtons = document.querySelectorAll('.tour-button');
    tourButtons.forEach(button => {
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            const tourId = this.getAttribute('data-tour-id');
            const tour = new GuidedTour(tourId);
            await tour.start();
        });
    });
});