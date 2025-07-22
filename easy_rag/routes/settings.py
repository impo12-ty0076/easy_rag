"""
Settings routes for Easy RAG System.

This module provides routes for managing system settings, including:
- .env file management for API keys
- Document storage path configuration
- Requirements.txt generation
- System diagnostics
"""
from flask import Blueprint, render_template, redirect, url_for, current_app, request, flash, jsonify, session
from easy_rag import db
from easy_rag.utils.dependency_manager import DependencyManager
from easy_rag.utils.diagnostics import SystemDiagnostics
import os
import platform
import sys
import subprocess
from dotenv import load_dotenv, set_key, find_dotenv
import shutil
import json
from datetime import datetime

bp = Blueprint('settings', __name__, url_prefix='/settings')

@bp.route('/', methods=['GET'])
def index():
    """Settings page"""
    # Load current environment variables
    dotenv_path = find_dotenv()
    env_vars = {}
    if dotenv_path:
        load_dotenv(dotenv_path)
        # Get relevant environment variables
        env_keys = ['OPENAI_API_KEY', 'HUGGINGFACE_API_KEY', 'PINECONE_API_KEY']
        env_vars = {key: os.environ.get(key, '') for key in env_keys}
    
    # Get system information
    system_info = DependencyManager.get_system_info()
    
    # Get dependency status
    core_deps = DependencyManager.check_core_dependencies()
    
    # Get installation history
    installation_history = DependencyManager.get_installation_history()
    
    return render_template(
        'settings/index.html', 
        env_vars=env_vars, 
        config=current_app.config,
        system_info=system_info,
        core_deps=core_deps,
        installation_history=installation_history
    )

@bp.route('/save', methods=['POST'])
def save():
    """Save settings"""
    # Update paths
    upload_folder = request.form.get('document_path')
    vector_db_folder = request.form.get('vector_db_path')
    
    # Create directories if they don't exist
    if upload_folder and upload_folder != current_app.config['UPLOAD_FOLDER']:
        os.makedirs(upload_folder, exist_ok=True)
        # Move existing files if needed
        if os.path.exists(current_app.config['UPLOAD_FOLDER']):
            for item in os.listdir(current_app.config['UPLOAD_FOLDER']):
                src = os.path.join(current_app.config['UPLOAD_FOLDER'], item)
                dst = os.path.join(upload_folder, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        current_app.config['UPLOAD_FOLDER'] = upload_folder
    
    if vector_db_folder and vector_db_folder != current_app.config['VECTOR_DB_FOLDER']:
        os.makedirs(vector_db_folder, exist_ok=True)
        # Move existing files if needed
        if os.path.exists(current_app.config['VECTOR_DB_FOLDER']):
            for item in os.listdir(current_app.config['VECTOR_DB_FOLDER']):
                src = os.path.join(current_app.config['VECTOR_DB_FOLDER'], item)
                dst = os.path.join(vector_db_folder, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        current_app.config['VECTOR_DB_FOLDER'] = vector_db_folder
    
    # Update max upload size
    max_upload_size = request.form.get('max_upload_size')
    if max_upload_size:
        current_app.config['MAX_CONTENT_LENGTH'] = int(max_upload_size) * 1024 * 1024
    
    # Update debug mode
    debug_mode = 'debug_mode' in request.form
    current_app.config['DEBUG'] = debug_mode
    
    # Update API keys in .env file
    dotenv_path = find_dotenv()
    if not dotenv_path:
        dotenv_path = os.path.join(os.getcwd(), '.env')
    
    api_keys = {
        'OPENAI_API_KEY': request.form.get('openai_api_key', ''),
        'HUGGINGFACE_API_KEY': request.form.get('huggingface_api_key', ''),
        'PINECONE_API_KEY': request.form.get('pinecone_api_key', '')
    }
    
    # Create .env file if it doesn't exist
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            f.write('# Environment variables for Easy RAG System\n')
    
    # Update .env file with API keys
    for key, value in api_keys.items():
        if value:  # Only update if value is provided
            set_key(dotenv_path, key, value)
    
    # Update document storage path in .env
    if upload_folder:
        set_key(dotenv_path, 'DOCUMENT_STORAGE_PATH', upload_folder)
    
    flash('Settings saved successfully', 'success')
    return redirect(url_for('settings.index'))

@bp.route('/generate-requirements')
def generate_requirements():
    """Generate requirements.txt file"""
    success = DependencyManager.generate_requirements_file('requirements.txt')
    
    if success:
        flash('requirements.txt file generated successfully', 'success')
    else:
        flash('Failed to generate requirements.txt file', 'error')
    
    return redirect(url_for('settings.index'))

@bp.route('/diagnostics')
def diagnostics():
    """System diagnostics page"""
    # Get system health information
    system_health = SystemDiagnostics.get_system_health()
    
    # Get dependency status
    core_deps = DependencyManager.check_core_dependencies()
    
    # Get installation history
    installation_history = DependencyManager.get_installation_history()
    
    # Get diagnostic information
    diagnostics = DependencyManager.diagnose_installation_issues()
    
    # Get error logs
    error_logs = _get_error_logs()
    
    return render_template(
        'settings/diagnostics.html',
        system_info=system_health['system'],
        resources=system_health['resources'],
        python_info=system_health['python'],
        database_health=system_health['database'],
        storage_health=system_health['storage'],
        health_status=system_health['status'],
        health_issues=system_health['issues'],
        core_deps=core_deps,
        installation_history=installation_history,
        diagnostics=diagnostics,
        error_logs=error_logs
    )

@bp.route('/diagnostics/run')
def run_diagnostics():
    """Run comprehensive system diagnostics"""
    # Run diagnostics
    diagnostic_results = SystemDiagnostics.run_diagnostics()
    
    # Store results in session for display
    if 'diagnostics_history' not in session:
        session['diagnostics_history'] = []
    
    # Add current diagnostic to history (keep only last 5)
    diagnostics_history = session['diagnostics_history']
    diagnostics_history.append({
        'timestamp': diagnostic_results['timestamp'],
        'execution_time': diagnostic_results['execution_time'],
        'health_status': diagnostic_results['health']['status']
    })
    
    # Keep only last 5 entries
    if len(diagnostics_history) > 5:
        diagnostics_history = diagnostics_history[-5:]
    
    session['diagnostics_history'] = diagnostics_history
    session['last_diagnostic'] = diagnostic_results
    
    flash('System diagnostics completed', 'success')
    return redirect(url_for('settings.diagnostics_results'))

@bp.route('/diagnostics/results')
def diagnostics_results():
    """Display results of the last diagnostic run"""
    # Get diagnostic results from session
    last_diagnostic = session.get('last_diagnostic')
    diagnostics_history = session.get('diagnostics_history', [])
    
    if not last_diagnostic:
        flash('No diagnostic results available. Please run diagnostics first.', 'warning')
        return redirect(url_for('settings.diagnostics'))
    
    return render_template(
        'settings/diagnostics_results.html',
        diagnostic=last_diagnostic,
        diagnostics_history=diagnostics_history
    )

@bp.route('/api/system-info')
def api_system_info():
    """API endpoint for system information"""
    system_info = DependencyManager.get_system_info()
    return jsonify(system_info)

@bp.route('/api/dependency-status')
def api_dependency_status():
    """API endpoint for dependency status"""
    core_deps = DependencyManager.check_core_dependencies()
    return jsonify(core_deps)

@bp.route('/api/installation-history')
def api_installation_history():
    """API endpoint for installation history"""
    installation_history = DependencyManager.get_installation_history()
    return jsonify(installation_history)

@bp.route('/api/diagnostics')
def api_diagnostics():
    """API endpoint for diagnostics"""
    diagnostics = DependencyManager.diagnose_installation_issues()
    return jsonify(diagnostics)

@bp.route('/api/system-health')
def api_system_health():
    """API endpoint for system health"""
    system_health = SystemDiagnostics.get_system_health()
    return jsonify(system_health)

@bp.route('/api/error-logs')
def api_error_logs():
    """API endpoint for error logs"""
    error_logs = _get_error_logs()
    return jsonify(error_logs)

def _get_error_logs(max_entries=50):
    """
    Get error logs from the log file.
    
    Args:
        max_entries (int): Maximum number of log entries to return
        
    Returns:
        list: List of error log entries
    """
    logs = []
    log_file = os.path.join('logs', 'easy_rag.log')
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
                # Process lines to extract error logs
                for line in lines[-1000:]:  # Look at last 1000 lines
                    if 'ERROR' in line:
                        logs.append(line.strip())
                        
                        # Limit the number of entries
                        if len(logs) >= max_entries:
                            break
        except Exception as e:
            logs.append(f"Error reading log file: {str(e)}")
    else:
        logs.append("Log file not found")
    
    return logs[-max_entries:]  # Return only the most recent entries