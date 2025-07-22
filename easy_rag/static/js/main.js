// Main JavaScript file for Easy RAG System

// Enable tooltips
document.addEventListener('DOMContentLoaded', function () {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Add fade-out effect to alerts after 5 seconds
    setTimeout(function () {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function (alert) {
            var bsAlert = new bootstrap.Alert(alert);
            setTimeout(function () {
                bsAlert.close();
            }, 5000);
        });
    }, 2000);

    // Load help components
    loadHelpComponents();
});

// Load help components
function loadHelpComponents() {
    // Load the help icon component
    const helpIconScript = document.createElement('script');
    helpIconScript.src = '/static/js/components/help-icon.js';
    document.head.appendChild(helpIconScript);

    // Add help button to navbar
    const navbarNav = document.querySelector('#navbarNav .navbar-nav');
    if (navbarNav) {
        const helpItem = document.createElement('li');
        helpItem.className = 'nav-item';
        helpItem.innerHTML = `
            <a class="nav-link" href="/help">
                <i class="bi bi-question-circle me-1"></i>Help
            </a>
        `;
        navbarNav.appendChild(helpItem);
    }
}