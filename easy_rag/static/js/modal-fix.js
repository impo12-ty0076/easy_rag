// Modal backdrop fix
document.addEventListener('DOMContentLoaded', function() {
    // Fix for modal backdrop not being removed
    document.body.addEventListener('hidden.bs.modal', function(event) {
        // Remove any lingering modal backdrops
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => {
            backdrop.remove();
        });
        
        // Remove modal-open class from body if no modals are open
        const openModals = document.querySelectorAll('.modal.show');
        if (openModals.length === 0) {
            document.body.classList.remove('modal-open');
        }
    });
});