"""
Routes for dependency management in the Easy RAG System.
"""
from flask import Blueprint, render_template, jsonify, request, current_app
from easy_rag.utils.dependency_manager import DependencyManager
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('dependency', __name__, url_prefix='/dependencies')

@bp.route('/')
def index():
    """Render the dependency management page."""
    # Get core dependencies status
    core_deps = DependencyManager.check_core_dependencies()
    
    # Get system information
    system_info = DependencyManager.get_system_info()
    
    # Get installation history
    installation_history = DependencyManager.get_installation_history()
    
    # Get core dependencies dictionary
    core_dependencies_dict = DependencyManager.CORE_DEPENDENCIES
    
    return render_template(
        'dependency/index.html',
        core_dependencies=core_deps,
        system_info=system_info,
        installation_history=installation_history,
        core_dependencies_dict=core_dependencies_dict
    )

@bp.route('/status')
def status():
    """Get the status of all dependencies."""
    # Get all dependencies
    all_deps = DependencyManager.get_all_dependencies()
    
    return jsonify(all_deps)

@bp.route('/core-status')
def core_status():
    """Get the status of core dependencies."""
    # Get core dependencies status
    core_deps = DependencyManager.check_core_dependencies()
    
    return jsonify(core_deps)

@bp.route('/feature-status/<feature>')
def feature_status(feature):
    """
    Get the status of dependencies for a specific feature.
    
    Args:
        feature (str): Feature name in format 'category/feature' (e.g., 'document_loaders/pdf')
    """
    # Get feature dependencies status
    feature_deps = DependencyManager.check_feature_dependencies(feature)
    
    return jsonify(feature_deps)

@bp.route('/install-core', methods=['POST'])
def install_core():
    """Install core dependencies."""
    # Start installation in a background thread
    current_app.executor.submit(DependencyManager.install_core_dependencies)
    
    return jsonify({
        "status": "started",
        "message": "Core dependency installation started"
    })

@bp.route('/install-feature', methods=['POST'])
def install_feature():
    """Install dependencies for a specific feature."""
    # Get feature from request
    feature = request.json.get('feature')
    if not feature:
        return jsonify({
            "status": "error",
            "message": "Feature not specified"
        }), 400
    
    # Start installation in a background thread
    current_app.executor.submit(DependencyManager.install_feature_dependencies, feature)
    
    return jsonify({
        "status": "started",
        "message": f"Feature dependency installation started for {feature}"
    })

@bp.route('/progress')
def progress():
    """Get the current progress of package installation."""
    # Get installation progress
    progress_data = DependencyManager.get_installation_progress()
    
    return jsonify(progress_data)

@bp.route('/cancel', methods=['POST'])
def cancel():
    """Cancel the current installation process."""
    # Cancel installation
    cancelled = DependencyManager.cancel_installation()
    
    if cancelled:
        return jsonify({
            "status": "success",
            "message": "Installation cancelled"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "No installation in progress"
        })

@bp.route('/diagnose')
def diagnose():
    """Diagnose installation issues."""
    # Get diagnostic information
    diagnostics = DependencyManager.diagnose_installation_issues()
    
    return jsonify(diagnostics)

@bp.route('/generate-requirements', methods=['POST'])
def generate_requirements():
    """Generate requirements.txt file."""
    # Get output path from request or use default
    output_path = request.json.get('output_path', 'requirements.txt')
    
    # Generate requirements file
    success = DependencyManager.generate_requirements_file(output_path)
    
    if success:
        return jsonify({
            "status": "success",
            "message": f"Requirements file generated at {output_path}"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to generate requirements file"
        }), 500

@bp.route('/features')
def features():
    """Get all available features with their dependencies."""
    features = {}
    
    # Get all features from DependencyManager
    for category, category_features in DependencyManager.FEATURE_DEPENDENCIES.items():
        for feature_name, deps in category_features.items():
            feature_id = f"{category}/{feature_name}"
            features[feature_id] = {
                "category": category,
                "name": feature_name,
                "dependencies": deps
            }
    
    return jsonify(features)