from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app
import os
import uuid
import json
from datetime import datetime
from easy_rag import db
from easy_rag.models import VectorDatabase, Document
from easy_rag.utils.document_loaders import get_available_loaders, AVAILABLE_LOADERS, get_loader_for_file
from easy_rag.utils.text_splitters import get_available_text_splitters_metadata, AVAILABLE_TEXT_SPLITTERS
from easy_rag.utils.embedding_models import (
    get_available_embedding_models, get_embedding_model,
    AVAILABLE_EMBEDDING_MODELS, get_available_vector_stores,
    install_vector_store_dependencies
)

bp = Blueprint('vector_db', __name__, url_prefix='/vector-db')

@bp.route('/')
def index():
    """List all vector databases"""
    vector_dbs = VectorDatabase.query.all()
    return render_template('vector_db/index.html', vector_dbs=vector_dbs)

@bp.route('/create', methods=['GET', 'POST'])
def create():
    """Create a new vector database - Step 1: Document Selection"""
    if request.method == 'POST':
        # Store form data in session for the multi-step process
        document_ids = request.form.getlist('document_ids')
        folder_path = request.form.get('folder_path')
        loader_ids = request.form.getlist('loader_ids')
        
        # Check if either documents or folder is selected
        if not document_ids and not folder_path:
            flash('Please select at least one document or a folder', 'danger')
            return redirect(url_for('vector_db.create'))
        
        # If no loaders are explicitly selected, automatically determine them
        if not loader_ids:
            # Get all file extensions from selected documents
            extensions = set()
            
            # Add extensions from selected documents
            if document_ids:
                documents = Document.query.filter(Document.id.in_(document_ids)).all()
                for doc in documents:
                    ext = os.path.splitext(doc.path)[1].lower()
                    if ext:
                        extensions.add(ext)
            
            # Add extensions from selected folder
            if folder_path:
                for root, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext:
                            extensions.add(ext)
            
            # Get appropriate loaders for detected extensions
            for ext in extensions:
                for loader_id, loader_class in AVAILABLE_LOADERS.items():
                    if ext in loader_class.supported_extensions:
                        is_available, _ = loader_class.check_dependencies()
                        if is_available and loader_id not in loader_ids:
                            loader_ids.append(loader_id)
        
        # Store selections in session
        session['vector_db_creation'] = {
            'document_ids': document_ids,
            'folder_path': folder_path,
            'loader_ids': loader_ids
        }
        
        # Redirect to text splitting configuration
        return redirect(url_for('vector_db.text_splitting'))
    
    # GET request - show the document and loader selection form
    documents = Document.query.all()
    
    # Get available folders in DOCUMENT_STORAGE_PATH
    upload_folder = current_app.config['UPLOAD_FOLDER']
    folders = []
    files = []
    
    # List all files and folders in the upload directory
    for item in os.listdir(upload_folder):
        item_path = os.path.join(upload_folder, item)
        if os.path.isdir(item_path):
            folders.append({
                'name': item,
                'path': item_path,
                'relative_path': os.path.relpath(item_path, upload_folder)
            })
        elif os.path.isfile(item_path):
            # Only include files that are not already in the database
            if not Document.query.filter_by(path=item_path).first():
                files.append({
                    'name': item,
                    'path': item_path,
                    'relative_path': os.path.relpath(item_path, upload_folder),
                    'size': os.path.getsize(item_path),
                    'extension': os.path.splitext(item)[1].lower()
                })
    
    # Get all file extensions in the upload directory for loader selection
    all_extensions = set()
    for root, _, filenames in os.walk(upload_folder):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext:  # Only add non-empty extensions
                all_extensions.add(ext)
    
    # Get loaders for the detected file extensions
    required_loaders = []
    for ext in all_extensions:
        for loader_id, loader_class in AVAILABLE_LOADERS.items():
            if ext in loader_class.supported_extensions and loader_id not in [l['id'] for l in required_loaders]:
                is_available, missing_packages = loader_class.check_dependencies()
                required_loaders.append({
                    'id': loader_id,
                    'name': loader_class.get_name(),
                    'description': loader_class.get_description(),
                    'supported_extensions': loader_class.get_supported_extensions(),
                    'required_packages': loader_class.get_required_packages(),
                    'is_available': is_available,
                    'missing_packages': missing_packages
                })
    return render_template('vector_db/create.html', 
                           documents=documents,
                           folders=folders,
                           files=files,
                           required_loaders=required_loaders)

@bp.route('/get_loaders')
def get_loaders():
    """API endpoint to get available document loaders"""
    # Get all available loaders
    loaders = get_available_loaders()
    
    # Check if we need to filter loaders based on detected extensions
    extensions = request.args.getlist('extensions')
    if extensions:
        filtered_loaders = []
        for loader in loaders:
            for ext in extensions:
                if ext in loader['supported_extensions']:
                    filtered_loaders.append(loader)
                    break
        return jsonify({'loaders': filtered_loaders})
    
    return jsonify({'loaders': loaders})

@bp.route('/get_text_splitters')
def get_text_splitters():
    """API endpoint to get available text splitters"""
    splitters = get_available_text_splitters_metadata()
    return jsonify({'splitters': splitters})

@bp.route('/get_loader_info')
def get_loader_info():
    """API endpoint to get information about a specific loader"""
    loader_id = request.args.get('loader_id')
    
    if not loader_id or loader_id not in AVAILABLE_LOADERS:
        return jsonify({'error': 'Invalid loader ID'}), 400
    
    loader_class = AVAILABLE_LOADERS[loader_id]
    is_available, missing_packages = loader_class.check_dependencies()
    
    return jsonify({
        'id': loader_id,
        'name': loader_class.get_name(),
        'description': loader_class.get_description(),
        'supported_extensions': loader_class.get_supported_extensions(),
        'required_packages': loader_class.get_required_packages(),
        'is_available': is_available,
        'missing_packages': missing_packages
    })

@bp.route('/install_dependencies', methods=['POST'])
def install_dependencies():
    """API endpoint to install dependencies for a loader"""
    data = request.get_json()
    loader_id = data.get('loader_id')
    
    if not loader_id or loader_id not in AVAILABLE_LOADERS:
        return jsonify({'success': False, 'error': 'Invalid loader ID'}), 400
    
    loader_class = AVAILABLE_LOADERS[loader_id]
    success, message = loader_class.install_dependencies()
    
    return jsonify({
        'success': success,
        'message': message
    })

@bp.route('/text-splitting', methods=['GET', 'POST'])
def text_splitting():
    """Step 2: Configure text splitting"""
    # Check if we have document selection data in session
    if 'vector_db_creation' not in session:
        flash('Please select documents and a loader first', 'warning')
        return redirect(url_for('vector_db.create'))
    
    if request.method == 'POST':
        # Get form data
        splitter_type = request.form.get('splitter_type')
        chunk_size = request.form.get('chunk_size')
        chunk_overlap = request.form.get('chunk_overlap')
        
        # Get separator parameters for character splitter
        separator_type = request.form.get('separator_type')
        custom_separator = request.form.get('custom_separator')
        is_separator_regex = request.form.get('is_separator_regex') == 'true'
        
        # Validate inputs
        if not splitter_type:
            flash('Please select a text splitting strategy', 'danger')
            return redirect(url_for('vector_db.text_splitting'))
        
        try:
            chunk_size = int(chunk_size)
            chunk_overlap = int(chunk_overlap)
            
            # Get the text splitter class
            from easy_rag.utils.text_splitters import AVAILABLE_TEXT_SPLITTERS
            
            if splitter_type not in AVAILABLE_TEXT_SPLITTERS or AVAILABLE_TEXT_SPLITTERS[splitter_type] is None:
                flash(f'Text splitter type "{splitter_type}" is not available yet', 'danger')
                return redirect(url_for('vector_db.text_splitting'))
                
            splitter_class = AVAILABLE_TEXT_SPLITTERS[splitter_type]
            
            # Handle separator for character splitter
            separator = None
            separators = None
            threshold_type = None
            threshold_amount = None
            
            if splitter_type == 'character':
                # Get separator based on type
                if separator_type == 'custom' and custom_separator:
                    separator = custom_separator
                elif separator_type in ['paragraph', 'line', 'sentence', 'comma', 'space']:
                    # Map separator type to actual separator
                    separator_map = {
                        'paragraph': '\n\n',
                        'line': '\n',
                        'sentence': '. ',
                        'comma': ', ',
                        'space': ' '
                    }
                    separator = separator_map.get(separator_type, '\n\n')
                else:
                    separator = '\n\n'  # Default separator
                
                # Validate parameters with separator
                is_valid, error_message = splitter_class.validate_parameters(
                    chunk_size, chunk_overlap, separator, is_separator_regex
                )
            elif splitter_type == 'recursive_character':
                # For RecursiveCharacterTextSplitter, use the default separators
                from easy_rag.utils.text_splitters import RecursiveCharacterTextSplitter
                separators = RecursiveCharacterTextSplitter.SEPARATOR_SETS["Default"]
                
                # Validate parameters with separators
                is_valid, error_message = splitter_class.validate_parameters(
                    chunk_size, chunk_overlap, separators, is_separator_regex
                )
            elif splitter_type == 'semantic':
                # For SemanticChunker, get threshold type and amount
                threshold_type = request.form.get('threshold_type', 'percentile')
                threshold_amount = float(chunk_size)  # Use chunk_size as threshold amount
                
                # We need an embedding model for semantic chunking
                # For now, we'll use a mock embedding model for validation
                # In a real implementation, you would use a proper embedding model
                class MockEmbeddingModel:
                    def embed_documents(self, texts):
                        return [[0.0] * 10 for _ in texts]  # Mock embeddings
                
                mock_model = MockEmbeddingModel()
                
                # Validate parameters with threshold type and amount
                is_valid, error_message = splitter_class.validate_parameters(
                    chunk_size, chunk_overlap, threshold_type, threshold_amount, '.', mock_model
                )
            else:
                # Validate parameters for other splitter types
                is_valid, error_message = splitter_class.validate_parameters(chunk_size, chunk_overlap)
                
            if not is_valid:
                flash(error_message, 'danger')
                return redirect(url_for('vector_db.text_splitting'))
                
        except ValueError:
            flash('Chunk size and overlap must be valid numbers', 'danger')
            return redirect(url_for('vector_db.text_splitting'))
        
        # Store text splitting configuration in session
        text_splitting_config = {
            'splitter_type': splitter_type,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
        # Add separator parameters for character splitter
        if splitter_type == 'character':
            text_splitting_config.update({
                'separator': separator,
                'is_separator_regex': is_separator_regex
            })
        # Add separator parameters for recursive character splitter
        elif splitter_type == 'recursive_character':
            text_splitting_config.update({
                'separators': separators,
                'is_separator_regex': is_separator_regex,
                'length_function': 'len'  # Store as string since functions can't be serialized
            })
        # Add parameters for code text splitter
        elif splitter_type == 'code':
            # Get the programming language from the form
            language = request.form.get('language', 'PYTHON')
            text_splitting_config.update({
                'language': language
            })
        # Add parameters for semantic chunker
        elif splitter_type == 'semantic':
            text_splitting_config.update({
                'breakpoint_threshold_type': threshold_type,
                'breakpoint_threshold_amount': threshold_amount,
                'sentence_separator': '.'
            })
            
        session['vector_db_creation'].update(text_splitting_config)
        
        # Redirect to the next step (embedding model selection)
        return redirect(url_for('vector_db.embedding_model'))
    
    # GET request - show the text splitting configuration form
    return render_template('vector_db/text_splitting.html')

@bp.route('/<id>')
def view(id):
    """View vector database details"""
    vector_db = VectorDatabase.query.get_or_404(id)
    
    # Get the documents included in this vector database
    documents = []
    if vector_db.document_ids:
        documents = Document.query.filter(Document.id.in_(vector_db.document_ids)).all()
    
    return render_template('vector_db/view.html', vector_db=vector_db, documents=documents)

@bp.route('/embedding-model', methods=['GET', 'POST'])
def embedding_model():
    """Step 3: Configure embedding model and vector store"""
    # Check if we have text splitting data in session
    if 'vector_db_creation' not in session:
        flash('Please configure text splitting first', 'warning')
        return redirect(url_for('vector_db.text_splitting'))
    
    if request.method == 'POST':
        # Get form data
        embedding_model_id = request.form.get('embedding_model_id')
        vector_store_id = request.form.get('vector_store_id')
        vector_db_name = request.form.get('vector_db_name')
        
        # Validate inputs
        if not embedding_model_id:
            flash('Please select an embedding model', 'danger')
            return redirect(url_for('vector_db.embedding_model'))
        
        if not vector_store_id:
            flash('Please select a vector store', 'danger')
            return redirect(url_for('vector_db.embedding_model'))
        
        if not vector_db_name:
            flash('Please enter a name for your vector database', 'danger')
            return redirect(url_for('vector_db.embedding_model'))
        
        # Store embedding model and vector store configuration in session
        session['vector_db_creation'].update({
            'embedding_model_id': embedding_model_id,
            'vector_store_id': vector_store_id,
            'vector_db_name': vector_db_name
        })
        
        # Create the vector database
        try:
            # Generate a unique ID for the vector database
            db_id = str(uuid.uuid4())
            
            # Create a path for the vector database files
            vector_db_path = os.path.join(os.getcwd(), 'instance', 'vector_dbs', db_id)
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Create the vector database record
            vector_db = VectorDatabase(
                id=db_id,
                name=vector_db_name,
                path=vector_db_path,
                created_at=datetime.now(),
                document_ids=session['vector_db_creation']['document_ids'],
                embedding_model=embedding_model_id,
                vector_store_type=vector_store_id,
                text_splitter={
                    'type': session['vector_db_creation']['splitter_type'],
                    'chunk_size': session['vector_db_creation']['chunk_size'],
                    'chunk_overlap': session['vector_db_creation']['chunk_overlap']
                },
                chunk_count=0,  # Will be updated during actual creation
                db_metadata={
                    'loader_id': session['vector_db_creation']['loader_id'],
                    'creation_status': 'processing',  # Will be updated during actual creation
                    'progress': 0,
                    'start_time': datetime.now().isoformat()
                }
            )
            
            db.session.add(vector_db)
            db.session.commit()
            
            # Clear the session data
            session.pop('vector_db_creation', None)
            
            # Redirect to the creation process page
            return redirect(url_for('vector_db.create_process', id=db_id))
            
        except Exception as e:
            flash(f'Error creating vector database: {str(e)}', 'danger')
            return redirect(url_for('vector_db.embedding_model'))
    
    # GET request - show the embedding model selection form
    return render_template('vector_db/embedding_model.html')

@bp.route('/get-embedding-models')
def get_embedding_models():
    """API endpoint to get available embedding models"""
    models = get_available_embedding_models()
    return jsonify({'models': models})

@bp.route('/get-vector-stores')
def get_vector_stores():
    """API endpoint to get available vector stores"""
    stores = get_available_vector_stores()
    return jsonify({'stores': stores})

@bp.route('/get-embedding-model-info')
def get_embedding_model_info():
    """API endpoint to get information about a specific embedding model"""
    model_id = request.args.get('model_id')
    
    if not model_id or model_id not in AVAILABLE_EMBEDDING_MODELS:
        return jsonify({'error': 'Invalid model ID'}), 400
    
    model_class = AVAILABLE_EMBEDDING_MODELS[model_id]
    is_available, missing_packages = model_class.check_dependencies()
    api_key_available = True
    api_key_error = ""
    
    if model_class.api_key_env:
        api_key_available, api_key_error = model_class.check_api_key()
    
    return jsonify({
        'id': model_id,
        'name': model_class.get_name(),
        'description': model_class.get_description(),
        'dimension': model_class.get_dimension(),
        'required_packages': model_class.get_required_packages(),
        'api_key_env': model_class.api_key_env,
        'is_available': is_available and api_key_available,
        'missing_packages': missing_packages,
        'api_key_error': api_key_error
    })

@bp.route('/get-vector-store-info')
def get_vector_store_info():
    """API endpoint to get information about a specific vector store"""
    store_id = request.args.get('store_id')
    
    if not store_id or store_id not in get_available_vector_stores():
        return jsonify({'error': 'Invalid vector store ID'}), 400
    
    stores = get_available_vector_stores()
    store_info = next((store for store in stores if store['id'] == store_id), None)
    
    if not store_info:
        return jsonify({'error': 'Vector store not found'}), 404
    
    return jsonify(store_info)

@bp.route('/install-embedding-model-dependencies', methods=['POST'])
def install_embedding_model_dependencies():
    """API endpoint to install dependencies for an embedding model"""
    data = request.get_json()
    model_id = data.get('model_id')
    
    if not model_id or model_id not in AVAILABLE_EMBEDDING_MODELS:
        return jsonify({'success': False, 'error': 'Invalid model ID'}), 400
    
    model_class = AVAILABLE_EMBEDDING_MODELS[model_id]
    success, message = model_class.install_dependencies()
    
    return jsonify({
        'success': success,
        'message': message
    })

@bp.route('/install-vector-store-dependencies', methods=['POST'])
def install_vector_store_dependencies_route():
    """API endpoint to install dependencies for a vector store"""
    data = request.get_json()
    store_id = data.get('store_id')
    
    success, message = install_vector_store_dependencies(store_id)
    
    return jsonify({
        'success': success,
        'message': message
    })

@bp.route('/<id>/create-process')
def create_process(id):
    """Show the vector database creation process page"""
    vector_db = VectorDatabase.query.get_or_404(id)
    
    # If the vector database is already created, redirect to the view page
    if vector_db.db_metadata.get('creation_status') == 'completed':
        flash('Vector database already created', 'info')
        return redirect(url_for('vector_db.view', id=id))
    
    # Start the creation process in a background thread
    if vector_db.db_metadata.get('creation_status') == 'processing' and vector_db.db_metadata.get('progress', 0) == 0:
        import threading
        thread = threading.Thread(target=process_vector_database_creation, args=(id,))
        thread.daemon = True
        thread.start()
    
    return render_template('vector_db/create_process.html', vector_db=vector_db)

@bp.route('/<id>/progress')
def get_creation_progress(id):
    """API endpoint to get the progress of vector database creation"""
    vector_db = VectorDatabase.query.get_or_404(id)
    
    return jsonify({
        'status': vector_db.db_metadata.get('creation_status', 'unknown'),
        'progress': vector_db.db_metadata.get('progress', 0),
        'current_operation': vector_db.db_metadata.get('current_operation', ''),
        'error': vector_db.db_metadata.get('error', '')
    })

def process_vector_database_creation(db_id):
    """Process the vector database creation in the background"""
    try:
        # Get the vector database
        vector_db = VectorDatabase.query.get(db_id)
        if not vector_db:
            return
        
        # Update metadata to indicate processing has started
        vector_db.db_metadata['creation_status'] = 'processing'
        vector_db.db_metadata['progress'] = 0
        vector_db.db_metadata['current_operation'] = 'Loading documents'
        db.session.commit()
        
        # Get the documents
        documents = Document.query.filter(Document.id.in_(vector_db.document_ids)).all()
        if not documents:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'No documents found'
            db.session.commit()
            return
        
        # Get the document loader
        loader_id = vector_db.db_metadata.get('loader_id')
        if not loader_id or loader_id not in AVAILABLE_LOADERS:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'Invalid document loader'
            db.session.commit()
            return
        
        # Update progress
        vector_db.db_metadata['progress'] = 10
        vector_db.db_metadata['current_operation'] = 'Loading document content'
        db.session.commit()
        
        # Load document content
        document_contents = []
        for i, document in enumerate(documents):
            try:
                # Get the appropriate loader for this document
                loader = get_loader_for_file(document.path)
                if not loader:
                    continue
                
                # Load the document
                doc_data = loader.load_document(document.path)
                document_contents.append(doc_data)
                
                # Update progress (10% to 30%)
                progress = 10 + int((i + 1) / len(documents) * 20)
                vector_db.db_metadata['progress'] = progress
                db.session.commit()
            except Exception as e:
                # Log the error but continue with other documents
                print(f"Error loading document {document.path}: {str(e)}")
        
        if not document_contents:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'Failed to load any documents'
            db.session.commit()
            return
        
        # Update progress
        vector_db.db_metadata['progress'] = 30
        vector_db.db_metadata['current_operation'] = 'Splitting text into chunks'
        db.session.commit()
        
        # Get the text splitter configuration
        splitter_type = vector_db.text_splitter.get('type')
        chunk_size = vector_db.text_splitter.get('chunk_size')
        chunk_overlap = vector_db.text_splitter.get('chunk_overlap')
        
        if not splitter_type or not chunk_size or not chunk_overlap:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'Invalid text splitter configuration'
            db.session.commit()
            return
        
        # Get the text splitter
        try:
            text_splitter = get_text_splitter(splitter_type)
        except Exception as e:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = f'Error getting text splitter: {str(e)}'
            db.session.commit()
            return
        
        # Split the text into chunks
        chunks = []
        chunk_metadata = []
        
        for i, doc_data in enumerate(document_contents):
            try:
                # Split the document content
                if splitter_type == 'character' and 'separator' in vector_db.text_splitter:
                    # For character splitter with custom separator
                    separator = vector_db.text_splitter.get('separator')
                    is_separator_regex = vector_db.text_splitter.get('is_separator_regex', False)
                    doc_chunks = text_splitter.split_text(
                        doc_data['content'], 
                        chunk_size, 
                        chunk_overlap,
                        separator,
                        is_separator_regex
                    )
                elif splitter_type == 'recursive_character' and 'separators' in vector_db.text_splitter:
                    # For recursive character splitter with custom separators
                    separators = vector_db.text_splitter.get('separators')
                    is_separator_regex = vector_db.text_splitter.get('is_separator_regex', False)
                    doc_chunks = text_splitter.split_text(
                        doc_data['content'], 
                        chunk_size, 
                        chunk_overlap,
                        separators,
                        is_separator_regex,
                        len  # Use len as the length function
                    )
                elif splitter_type == 'semantic' and 'breakpoint_threshold_type' in vector_db.text_splitter:
                    # For semantic chunker
                    breakpoint_threshold_type = vector_db.text_splitter.get('breakpoint_threshold_type')
                    breakpoint_threshold_amount = vector_db.text_splitter.get('breakpoint_threshold_amount')
                    sentence_separator = vector_db.text_splitter.get('sentence_separator', '.')
                    
                    # Create a simple embedding model for demonstration
                    # In a real implementation, you would use a proper embedding model
                    class SimpleEmbeddingModel:
                        def embed_documents(self, texts):
                            # This is a very simplified embedding that just uses the length and first character
                            # In a real implementation, you would use a proper embedding model
                            import numpy as np
                            return [np.array([len(t), ord(t[0]) if t else 0]) for t in texts]
                    
                    embedding_model = SimpleEmbeddingModel()
                    
                    doc_chunks = text_splitter.split_text(
                        doc_data['content'],
                        chunk_size,
                        chunk_overlap,
                        breakpoint_threshold_type,
                        breakpoint_threshold_amount,
                        sentence_separator,
                        embedding_model
                    )
                else:
                    # For other splitters
                    doc_chunks = text_splitter.split_text(
                        doc_data['content'], 
                        chunk_size, 
                        chunk_overlap
                    )
                
                # Add metadata to each chunk
                for chunk in doc_chunks:
                    chunks.append(chunk)
                    chunk_metadata.append({
                        'source': doc_data['metadata'].get('source', ''),
                        'document_id': documents[i].id,
                        'document_name': documents[i].name,
                        'chunk_index': len(chunk_metadata)
                    })
                
                # Update progress (30% to 50%)
                progress = 30 + int((i + 1) / len(document_contents) * 20)
                vector_db.db_metadata['progress'] = progress
                db.session.commit()
            except Exception as e:
                # Log the error but continue with other documents
                print(f"Error splitting document: {str(e)}")
        
        if not chunks:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'Failed to create any text chunks'
            db.session.commit()
            return
        
        # Update progress
        vector_db.db_metadata['progress'] = 50
        vector_db.db_metadata['current_operation'] = 'Creating embeddings'
        db.session.commit()
        
        # Get the embedding model
        embedding_model_id = vector_db.embedding_model
        if not embedding_model_id or embedding_model_id not in AVAILABLE_EMBEDDING_MODELS:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = 'Invalid embedding model'
            db.session.commit()
            return
        
        try:
            embedding_model = get_embedding_model(embedding_model_id)
        except Exception as e:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = f'Error getting embedding model: {str(e)}'
            db.session.commit()
            return
        
        # Create embeddings
        try:
            # Process in batches to avoid memory issues
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_embeddings = embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress (50% to 80%)
                progress = 50 + int((i + len(batch)) / len(chunks) * 30)
                vector_db.db_metadata['progress'] = progress
                db.session.commit()
        except Exception as e:
            vector_db.db_metadata['creation_status'] = 'error'
            vector_db.db_metadata['error'] = f'Error creating embeddings: {str(e)}'
            db.session.commit()
            return
        
        # Update progress
        vector_db.db_metadata['progress'] = 80
        vector_db.db_metadata['current_operation'] = 'Creating vector store'
        db.session.commit()
        
        # Get the vector store type
        vector_store_type = vector_db.vector_store_type
        if not vector_store_type:
            vector_db.metadata['creation_status'] = 'error'
            vector_db.metadata['error'] = 'Invalid vector store type'
            db.session.commit()
            return
        
        # Create the vector store
        try:
            if vector_store_type == 'chroma':
                import chromadb
                from chromadb.config import Settings
                
                # Create a Chroma client
                client = chromadb.Client(Settings(
                    persist_directory=vector_db.path,
                    anonymized_telemetry=False
                ))
                
                # Create a collection
                collection = client.create_collection(
                    name=vector_db.name,
                    metadata={"description": f"Vector database for {vector_db.name}"}
                )
                
                # Add documents to the collection
                collection.add(
                    embeddings=all_embeddings,
                    documents=chunks,
                    metadatas=chunk_metadata,
                    ids=[f"chunk_{i}" for i in range(len(chunks))]
                )
                
            elif vector_store_type == 'faiss':
                import faiss
                import numpy as np
                import pickle
                
                # Convert embeddings to numpy array
                embeddings_array = np.array(all_embeddings).astype('float32')
                
                # Get the dimension of the embeddings
                dimension = len(all_embeddings[0])
                
                # Create a FAISS index
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_array)
                
                # Save the index
                faiss.write_index(index, os.path.join(vector_db.path, 'index.faiss'))
                
                # Save the documents and metadata
                with open(os.path.join(vector_db.path, 'documents.pkl'), 'wb') as f:
                    pickle.dump({'documents': chunks, 'metadata': chunk_metadata}, f)
            
            else:
                vector_db.metadata['creation_status'] = 'error'
                vector_db.metadata['error'] = f'Unsupported vector store type: {vector_store_type}'
                db.session.commit()
                return
            
        except Exception as e:
            vector_db.metadata['creation_status'] = 'error'
            vector_db.metadata['error'] = f'Error creating vector store: {str(e)}'
            db.session.commit()
            return
        
        # Update the vector database record
        vector_db.chunk_count = len(chunks)
        vector_db.metadata['creation_status'] = 'completed'
        vector_db.metadata['progress'] = 100
        vector_db.metadata['current_operation'] = 'Vector database creation complete'
        vector_db.metadata['completion_time'] = datetime.now().isoformat()
        vector_db.metadata['chunk_count'] = len(chunks)
        vector_db.metadata['embedding_dimension'] = len(all_embeddings[0]) if all_embeddings else 0
        db.session.commit()
        
    except Exception as e:
        # Handle any unexpected errors
        try:
            vector_db = VectorDatabase.query.get(db_id)
            if vector_db:
                vector_db.metadata['creation_status'] = 'error'
                vector_db.metadata['error'] = f'Unexpected error: {str(e)}'
                db.session.commit()
        except:
            pass

@bp.route('/<id>/delete', methods=['POST'])
def delete(id):
    """Delete a vector database"""
    vector_db = VectorDatabase.query.get_or_404(id)
    
    # Delete the vector database files
    try:
        if os.path.exists(vector_db.path):
            import shutil
            shutil.rmtree(vector_db.path)
    except Exception as e:
        flash(f'Error deleting vector database files: {str(e)}', 'warning')
    
    # Delete the database record
    db.session.delete(vector_db)
    db.session.commit()
    
    flash(f'Vector database {vector_db.name} deleted successfully', 'success')
    return redirect(url_for('vector_db.index'))