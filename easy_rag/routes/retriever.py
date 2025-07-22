from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from easy_rag import db
import json
import uuid
from easy_rag.models import VectorDatabase, Configuration
from easy_rag.utils.retrievers import (
    get_retriever_types, 
    get_retriever_type, 
    validate_retriever_parameters,
    get_default_parameters,
    get_reranking_llms,
    get_reranking_llm,
    check_api_key_availability,
    validate_advanced_retrieval_options
)

bp = Blueprint('retriever', __name__, url_prefix='/retriever')

@bp.route('/')
def index():
    """Configure retriever settings"""
    vector_dbs = VectorDatabase.query.all()
    
    # Get configurations for all vector databases
    configs = {}
    for db in vector_dbs:
        config = Configuration.query.filter_by(name=f"retriever_{db.id}").first()
        if config and config.settings and 'retriever_type' in config.settings:
            retriever_type = config.settings['retriever_type']
            retriever_info = get_retriever_type(retriever_type)
            if retriever_info:
                configs[db.id] = {
                    'name': retriever_info['name'],
                    'icon': retriever_info.get('icon', 'search'),
                    'type': retriever_type
                }
    
    return render_template('retriever/index.html', vector_dbs=vector_dbs, configs=configs)

@bp.route('/configure/<db_id>', methods=['GET', 'POST'])
def configure(db_id):
    """Configure retriever for a specific vector database"""
    vector_db = VectorDatabase.query.get_or_404(db_id)
    retriever_types = get_retriever_types()
    reranking_llms = get_reranking_llms()
    
    # Group retriever types by category
    categorized_retrievers = {}
    for r_id, r_info in retriever_types.items():
        category = r_info.get('category', 'other')
        if category not in categorized_retrievers:
            categorized_retrievers[category] = {}
        categorized_retrievers[category][r_id] = r_info
    
    # Check if there's an existing configuration for this vector DB
    config = Configuration.query.filter_by(name=f"retriever_{db_id}").first()
    current_config = config.settings if config else {}
    
    # Check API key availability for reranking LLMs
    for llm_id, llm_info in reranking_llms.items():
        if llm_info.get('requires_api_key', False):
            api_key_name = llm_info.get('api_key_name')
            if api_key_name:
                llm_info['api_key_available'] = check_api_key_availability(api_key_name)
            else:
                llm_info['api_key_available'] = False
        else:
            llm_info['api_key_available'] = True
    
    if request.method == 'POST':
        retriever_type = request.form.get('retriever_type')
        
        if not retriever_type or retriever_type not in retriever_types:
            flash('Please select a valid retriever type', 'error')
            return render_template(
                'retriever/configure.html', 
                vector_db=vector_db, 
                retriever_types=retriever_types,
                categorized_retrievers=categorized_retrievers,
                reranking_llms=reranking_llms,
                current_config=current_config
            )
        
        # Get parameters for the selected retriever type
        parameters = {}
        for param_name, param_config in retriever_types[retriever_type]['parameters'].items():
            # Handle different parameter types
            if param_config['type'] == 'boolean':
                # Checkboxes only send a value when checked
                parameters[param_name] = request.form.get(param_name) == 'on'
            else:
                value = request.form.get(param_name)
                if value is not None and value.strip() != '':
                    parameters[param_name] = value
        
        # Validate parameters
        errors = validate_retriever_parameters(retriever_type, parameters)
        
        # Get advanced retrieval options
        advanced_options = {
            'reranking_llm': request.form.get('reranking_llm', 'none'),
            'chunk_count': request.form.get('chunk_count', '4'),
            'use_hybrid_search': request.form.get('use_hybrid_search') == 'on',
        }
        
        # Add hybrid search alpha if hybrid search is enabled
        if advanced_options['use_hybrid_search']:
            advanced_options['hybrid_alpha'] = request.form.get('hybrid_alpha', '0.5')
        
        # Validate advanced options
        advanced_errors = validate_advanced_retrieval_options(advanced_options)
        errors.update(advanced_errors)
        
        if errors:
            for field, error in errors.items():
                flash(f"{field}: {error}", 'error')
            return render_template(
                'retriever/configure.html', 
                vector_db=vector_db, 
                retriever_types=retriever_types,
                categorized_retrievers=categorized_retrievers,
                reranking_llms=reranking_llms,
                selected_type=retriever_type,
                parameters=parameters,
                advanced_options=advanced_options,
                current_config=current_config,
                errors=errors
            )
        
        # Save configuration
        config_data = {
            'retriever_type': retriever_type,
            'parameters': parameters,
            'advanced_options': advanced_options,
            'updated_at': db.func.now().isoformat() if hasattr(db.func.now(), 'isoformat') else str(db.func.now())
        }
        
        if config:
            config.settings = config_data
            config.updated_at = db.func.now()
        else:
            config = Configuration(
                id=str(uuid.uuid4()),
                name=f"retriever_{db_id}",
                settings=config_data
            )
            db.session.add(config)
        
        db.session.commit()
        flash('Retriever configuration saved successfully', 'success')
        return redirect(url_for('retriever.configure', db_id=db_id))
    
    return render_template(
        'retriever/configure.html', 
        vector_db=vector_db, 
        retriever_types=retriever_types,
        categorized_retrievers=categorized_retrievers,
        reranking_llms=reranking_llms,
        current_config=current_config
    )

@bp.route('/parameters/<retriever_type>')
def get_parameters(retriever_type):
    """Get parameters for a specific retriever type"""
    retriever_config = get_retriever_type(retriever_type)
    
    if not retriever_config:
        return jsonify({'error': 'Invalid retriever type'}), 400
    
    return jsonify({
        'parameters': retriever_config['parameters'],
        'defaults': get_default_parameters(retriever_type)
    })

@bp.route('/parameters-for-db/<db_id>')
def get_parameters_for_db(db_id):
    """Get retriever configuration for a specific vector database"""
    # Check if there's an existing configuration for this vector DB
    config = Configuration.query.filter_by(name=f"retriever_{db_id}").first()
    
    if not config or not config.settings:
        # Return default configuration
        return jsonify({
            'retriever_type': 'similarity',
            'retriever_name': 'Similarity Search',
            'parameters': get_default_parameters('similarity'),
            'advanced_options': {
                'chunk_count': 4,
                'reranking_llm': 'none',
                'use_hybrid_search': False
            }
        })
    
    # Get retriever type info
    retriever_type = config.settings.get('retriever_type', 'similarity')
    retriever_info = get_retriever_type(retriever_type)
    retriever_name = retriever_info.get('name', 'Similarity Search') if retriever_info else 'Similarity Search'
    
    # Get reranking LLM info if applicable
    reranking_llm = None
    reranking_name = 'No Reranking'
    
    if 'advanced_options' in config.settings and 'reranking_llm' in config.settings['advanced_options']:
        reranking_llm = config.settings['advanced_options']['reranking_llm']
        if reranking_llm != 'none':
            reranking_info = get_reranking_llm(reranking_llm)
            if reranking_info:
                reranking_name = reranking_info.get('name', 'Reranking Enabled')
    
    # Return the configuration
    return jsonify({
        'retriever_type': retriever_type,
        'retriever_name': retriever_name,
        'parameters': config.settings.get('parameters', {}),
        'advanced_options': config.settings.get('advanced_options', {
            'chunk_count': 4,
            'reranking_llm': 'none',
            'use_hybrid_search': False
        }),
        'reranking_name': reranking_name
    })