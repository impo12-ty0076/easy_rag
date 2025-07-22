"""
Routes for dependency management in the Easy RAG System.
This module provides API endpoints for checking, installing, and managing dependencies.
"""
from flask import Blueprint, jsonify, request, render_template, current_app, abort
from easy_rag.utils.dependency_manager import DependencyManager
import os
import time

bp = Blueprint('dependencies', __name__, url_prefix='/dependencies')

@bp.route('/check/core', methods=['GET'])
def check_core_dependencies():
    """Check core dependencies and return status."""
    dependencies = DependencyManager.check_core_dependencies()
    return jsonify({
        'dependencies': dependencies,
        'all_installed': all(dependencies.values())
    })

@bp.route('/check/feature/<path:feature>', methods=['GET'])
def check_feature_dependencies(feature):
    """Check feature dependencies and return status."""
    dependencies = DependencyManager.check_feature_dependencies(feature)
    if not dependencies:
        return jsonify({'error': f'Invalid feature: {feature}'}), 400
    
    return jsonify({
        'feature': feature,
        'dependencies': dependencies,
        'all_installed': all(dependencies.values())
    })

@bp.route('/install/core', methods=['POST'])
def install_core_dependencies():
    """Install core dependencies."""
    # Start installation in a background thread
    current_app.task_queue.enqueue(DependencyManager.install_core_dependencies)
    return jsonify({'status': 'installation_started'})

@bp.route('/install/feature/<path:feature>', methods=['POST'])
def install_feature_dependencies(feature):
    """Install feature dependencies."""
    # Start installation in a background thread
    current_app.task_queue.enqueue(
        DependencyManager.install_feature_dependencies,
        feature
    )
    return jsonify({'status': 'installation_started', 'feature': feature})

@bp.route('/progress', methods=['GET'])
def get_installation_progress():
    """Get the current progress of dependency installation."""
    progress = DependencyManager.get_installation_progress()
    return jsonify(progress)

@bp.route('/generate-requirements', methods=['POST'])
def generate_requirements():
    """Generate requirements.txt file."""
    output_path = request.json.get('output_path', 'requirements.txt')
    success = DependencyManager.generate_requirements_file(output_path)
    
    if success:
        return jsonify({'status': 'success', 'path': output_path})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to generate requirements file'}), 500

@bp.route('/cancel', methods=['POST'])
def cancel_installation():
    """Cancel the current dependency installation."""
    cancelled = DependencyManager.cancel_installation()
    return jsonify({'cancelled': cancelled})

@bp.route('/history', methods=['GET'])
def get_installation_history():
    """Get the installation history."""
    history = DependencyManager.get_installation_history()
    return jsonify({'history': history})

@bp.route('/system-info', methods=['GET'])
def get_system_info():
    """Get system information."""
    system_info = DependencyManager.get_system_info()
    return jsonify(system_info)

@bp.route('/details/<package_name>', methods=['GET'])
def get_dependency_details(package_name):
    """Get detailed information about a dependency."""
    details = DependencyManager.get_dependency_details(package_name)
    return jsonify(details)

@bp.route('/all', methods=['GET'])
def get_all_dependencies():
    """Get details for all dependencies."""
    all_deps = DependencyManager.get_all_dependencies()
    return jsonify(all_deps)

@bp.route('/diagnose', methods=['GET'])
def diagnose_installation_issues():
    """Diagnose common installation issues."""
    diagnostics = DependencyManager.diagnose_installation_issues()
    return jsonify(diagnostics)

@bp.route('/dashboard', methods=['GET'])
def dependency_dashboard():
    """Render dependency management dashboard."""
    core_deps = DependencyManager.check_core_dependencies()
    
    # Get feature categories
    feature_categories = {}
    for category, features in DependencyManager.FEATURE_DEPENDENCIES.items():
        feature_categories[category] = list(features.keys())
    
    # Get system information
    system_info = DependencyManager.get_system_info()
    
    return render_template(
        'dependencies/dashboard.html',
        core_dependencies=core_deps,
        feature_categories=feature_categories,
        system_info=system_info
    )