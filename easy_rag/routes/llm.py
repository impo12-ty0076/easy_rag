from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from easy_rag import db
import json
import uuid
import os
from easy_rag.models import Configuration
from easy_rag.utils.llms import (
    get_available_llms,
    get_llm_info,
    check_api_key_availability,
    is_model_downloaded,
    download_model,
    get_llm_availability
)

bp = Blueprint('llm', __name__, url_prefix='/llm')

@bp.route('/')
def index():
    """LLM selection interface"""
    # Get all available LLMs
    llms = get_available_llms()
    
    # Group LLMs by category
    categorized_llms = {}
    for llm_id, llm_info in llms.items():
        category = llm_info.get('category', 'other')
        if category not in categorized_llms:
            categorized_llms[category] = {}
        categorized_llms[category][llm_id] = llm_info
    
    # Check availability of all LLMs
    availability = get_llm_availability()
    
    # Get current LLM configuration if it exists
    config = Configuration.query.filter_by(name="llm_config").first()
    current_config = config.settings if config else {}
    
    return render_template(
        'llm/index.html',
        llms=llms,
        categorized_llms=categorized_llms,
        availability=availability,
        current_config=current_config
    )

@bp.route('/configure', methods=['POST'])
def configure():
    """Configure LLM settings"""
    llm_id = request.form.get('llm_id')
    
    if not llm_id:
        flash('Please select an LLM', 'error')
        return redirect(url_for('llm.index'))
    
    # Get LLM info
    llm_info = get_llm_info(llm_id)
    if not llm_info:
        flash('Invalid LLM selection', 'error')
        return redirect(url_for('llm.index'))
    
    # Check if API key is required and available
    if llm_info.get('requires_api_key', False):
        api_key_name = llm_info.get('api_key_name')
        if api_key_name and not check_api_key_availability(api_key_name):
            flash(f'API key not found for {llm_info["name"]}. Please add {api_key_name} to your .env file.', 'error')
            return redirect(url_for('llm.index'))
    
    # Check if model download is required
    if llm_info.get('download_required', False):
        model_id = llm_info.get('model_id')
        quantization = llm_info.get('quantization')
        
        if model_id and not is_model_downloaded(model_id, quantization):
            # Model needs to be downloaded
            flash(f'Downloading {llm_info["name"]}. This may take some time...', 'info')
            
            # In a real implementation, this would be a background task
            success = download_model(model_id, quantization)
            
            if not success:
                flash(f'Failed to download {llm_info["name"]}', 'error')
                return redirect(url_for('llm.index'))
            
            flash(f'{llm_info["name"]} downloaded successfully', 'success')
    
    # Save LLM configuration
    config_data = {
        'llm_id': llm_id,
        'name': llm_info['name'],
        'updated_at': db.func.now().isoformat() if hasattr(db.func.now(), 'isoformat') else str(db.func.now())
    }
    
    # Get existing config or create new one
    config = Configuration.query.filter_by(name="llm_config").first()
    
    if config:
        config.settings = config_data
        config.updated_at = db.func.now()
    else:
        config = Configuration(
            id=str(uuid.uuid4()),
            name="llm_config",
            settings=config_data
        )
        db.session.add(config)
    
    db.session.commit()
    
    flash(f'{llm_info["name"]} selected successfully', 'success')
    return redirect(url_for('query.index'))

@bp.route('/check_api_key', methods=['POST'])
def check_api_key():
    """Check if an API key is available"""
    api_key_name = request.json.get('api_key_name')
    
    if not api_key_name:
        return jsonify({'error': 'API key name is required'}), 400
    
    available = check_api_key_availability(api_key_name)
    
    return jsonify({
        'available': available,
        'api_key_name': api_key_name
    })

@bp.route('/download_status/<model_id>')
def download_status(model_id):
    """Check the download status of a model"""
    quantization = request.args.get('quantization')
    
    # In a real implementation, this would check the actual download status
    # For now, we'll just return a placeholder status
    downloaded = is_model_downloaded(model_id, quantization)
    
    return jsonify({
        'model_id': model_id,
        'downloaded': downloaded,
        'progress': 100 if downloaded else 0
    })