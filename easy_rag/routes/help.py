"""
Help routes for Easy RAG System

This module provides routes for the help system, including tooltips, guided tours,
and error explanations.
"""
from flask import Blueprint, jsonify, render_template, current_app
import os
import json

bp = Blueprint('help', __name__, url_prefix='/help')

# Dictionary of error explanations
ERROR_EXPLANATIONS = {
    'db_connection': {
        'title': 'Database Connection Error',
        'explanation': 'The application could not connect to the database. This could be due to a corrupted database file or insufficient permissions.',
        'solution': 'Try restarting the application. If the problem persists, check that the database file exists and has the correct permissions. You may need to recreate the database using the init_db.py script.'
    },
    'file_upload': {
        'title': 'File Upload Error',
        'explanation': 'The file could not be uploaded. This could be due to an unsupported file format, a file that is too large, or insufficient permissions.',
        'solution': 'Check that the file is in a supported format (PDF, TXT, DOCX, etc.) and is smaller than the maximum allowed size (16MB). Also, ensure that the uploads directory exists and has the correct permissions.'
    },
    'vector_db_creation': {
        'title': 'Vector Database Creation Error',
        'explanation': 'The vector database could not be created. This could be due to issues with the document content, embedding model, or vector store.',
        'solution': 'Check that the documents are valid and contain text that can be extracted. Ensure that the selected embedding model is available and that you have provided any required API keys. Try using a different vector store or embedding model.'
    },
    'dependency_installation': {
        'title': 'Dependency Installation Error',
        'explanation': 'The required dependencies could not be installed. This could be due to network issues, incompatible versions, or insufficient permissions.',
        'solution': 'Check your internet connection and try again. If the problem persists, try installing the dependencies manually using pip. You can find the list of required dependencies in the requirements.txt file.'
    },
    'api_key': {
        'title': 'API Key Error',
        'explanation': 'The application could not authenticate with the API. This could be due to an invalid or missing API key.',
        'solution': 'Check that you have provided the correct API key in the settings. If you don\'t have an API key, you can obtain one from the service provider\'s website.'
    },
    'model_download': {
        'title': 'Model Download Error',
        'explanation': 'The model could not be downloaded. This could be due to network issues, insufficient disk space, or incompatible model versions.',
        'solution': 'Check your internet connection and available disk space. Try downloading a smaller model or using an API-based model instead.'
    },
    'query_processing': {
        'title': 'Query Processing Error',
        'explanation': 'The query could not be processed. This could be due to issues with the vector database, retriever, or language model.',
        'solution': 'Check that the vector database exists and contains documents. Ensure that the selected retriever and language model are available and properly configured. Try using different retriever settings or a different language model.'
    }
}

# Dictionary of help content
HELP_CONTENT = {
    'document_management': {
        'title': 'Document Management',
        'content': '''
            <h4>Managing Your Documents</h4>
            <p>The Document Management section allows you to upload, view, and delete documents that will be used in your RAG system.</p>
            
            <h5>Uploading Documents</h5>
            <p>To upload documents, you can either:</p>
            <ul>
                <li>Click the "Upload Documents" button and select files from your computer</li>
                <li>Drag and drop files directly onto the upload area</li>
            </ul>
            <p>Supported file formats include PDF, TXT, DOCX, HTML, and more.</p>
            
            <h5>Managing Documents</h5>
            <p>For each document, you can:</p>
            <ul>
                <li>View document details and preview content</li>
                <li>Delete documents you no longer need</li>
            </ul>
            
            <h5>Tips</h5>
            <ul>
                <li>Organize related documents in folders before uploading</li>
                <li>Use descriptive filenames to easily identify documents</li>
                <li>Preview documents to ensure content is extracted correctly</li>
            </ul>
        '''
    },
    'vector_db_creation': {
        'title': 'Vector Database Creation',
        'content': '''
            <h4>Creating Vector Databases</h4>
            <p>Vector databases store your documents as numerical representations (embeddings) that enable semantic search.</p>
            
            <h5>Creating a Vector Database</h5>
            <ol>
                <li>Select the documents you want to include</li>
                <li>Choose a document loader appropriate for your file types</li>
                <li>Configure text splitting settings to control how documents are chunked</li>
                <li>Select an embedding model to convert text to vectors</li>
                <li>Choose a vector store to save the embeddings</li>
            </ol>
            
            <h5>Text Splitting</h5>
            <p>Text splitting divides documents into smaller chunks for more precise retrieval:</p>
            <ul>
                <li><strong>Chunk Size</strong>: The number of characters or tokens in each chunk</li>
                <li><strong>Chunk Overlap</strong>: The number of characters or tokens that overlap between adjacent chunks</li>
            </ul>
            
            <h5>Embedding Models</h5>
            <p>Different embedding models offer different trade-offs between quality and speed:</p>
            <ul>
                <li><strong>Local Models</strong>: Run on your computer, no API key required</li>
                <li><strong>API-based Models</strong>: Higher quality but require an API key and internet connection</li>
            </ul>
            
            <h5>Vector Stores</h5>
            <p>Vector stores save and index your embeddings for efficient retrieval:</p>
            <ul>
                <li><strong>Chroma</strong>: Good all-around performance</li>
                <li><strong>FAISS</strong>: Optimized for large collections</li>
                <li><strong>Pinecone</strong>: Cloud-based, requires API key</li>
            </ul>
        '''
    },
    'retriever_configuration': {
        'title': 'Retriever Configuration',
        'content': '''
            <h4>Configuring Retrievers</h4>
            <p>Retrievers determine how documents are fetched from the vector database when you ask a question.</p>
            
            <h5>Retriever Types</h5>
            <ul>
                <li><strong>Similarity Search</strong>: Basic semantic search using vector similarity</li>
                <li><strong>MMR (Maximum Marginal Relevance)</strong>: Balances relevance with diversity in results</li>
                <li><strong>Hybrid Search</strong>: Combines semantic search with keyword search</li>
            </ul>
            
            <h5>Configuration Options</h5>
            <ul>
                <li><strong>Number of Chunks</strong>: How many document chunks to retrieve for each query</li>
                <li><strong>Reranking</strong>: Optional second-pass ranking to improve relevance</li>
                <li><strong>Score Threshold</strong>: Minimum similarity score for retrieved chunks</li>
            </ul>
            
            <h5>Tips for Effective Retrieval</h5>
            <ul>
                <li>Start with Similarity Search for most use cases</li>
                <li>Use MMR when you want diverse perspectives</li>
                <li>Try Hybrid Search for technical content with specific terminology</li>
                <li>Adjust the number of chunks based on your needs (more chunks = more context but potentially more noise)</li>
            </ul>
        '''
    },
    'query_interface': {
        'title': 'Query Interface',
        'content': '''
            <h4>Querying Your RAG System</h4>
            <p>The Query Interface allows you to ask questions and get answers based on your documents.</p>
            
            <h5>Using the Query Interface</h5>
            <ol>
                <li>Select a vector database to query</li>
                <li>Configure the retriever settings</li>
                <li>Choose a language model (LLM)</li>
                <li>Enter your question in the query box</li>
                <li>Click "Submit" to get an answer</li>
            </ol>
            
            <h5>Language Models</h5>
            <p>Different language models offer different capabilities:</p>
            <ul>
                <li><strong>Local Models</strong>: Run on your computer, no API key required, but may be slower or less capable</li>
                <li><strong>API-based Models</strong>: Higher quality but require an API key and internet connection</li>
            </ul>
            
            <h5>Understanding Results</h5>
            <p>The query results include:</p>
            <ul>
                <li><strong>Answer</strong>: The generated response to your question</li>
                <li><strong>Source Chunks</strong>: The document chunks that were used to generate the answer</li>
                <li><strong>Relevance Scores</strong>: How relevant each chunk is to your question</li>
            </ul>
            
            <h5>Tips for Better Questions</h5>
            <ul>
                <li>Be specific in your questions</li>
                <li>Use clear, concise language</li>
                <li>Break complex questions into simpler ones</li>
                <li>Experiment with different retriever settings for different types of questions</li>
            </ul>
        '''
    },
    'settings': {
        'title': 'System Settings',
        'content': '''
            <h4>Configuring System Settings</h4>
            <p>The Settings page allows you to configure various aspects of the Easy RAG System.</p>
            
            <h5>API Keys</h5>
            <p>Some features require API keys from external services:</p>
            <ul>
                <li><strong>OpenAI API Key</strong>: Required for OpenAI models like GPT-4</li>
                <li><strong>Hugging Face API Key</strong>: Required for some Hugging Face models</li>
                <li><strong>Pinecone API Key</strong>: Required for Pinecone vector store</li>
            </ul>
            <p>API keys are stored securely in your .env file.</p>
            
            <h5>Storage Paths</h5>
            <p>You can configure where documents and vector databases are stored:</p>
            <ul>
                <li><strong>Document Storage Path</strong>: Where uploaded documents are saved</li>
                <li><strong>Vector Database Path</strong>: Where vector databases are saved</li>
            </ul>
            
            <h5>Dependencies</h5>
            <p>You can manage system dependencies:</p>
            <ul>
                <li>View installed dependencies</li>
                <li>Install missing dependencies</li>
                <li>Generate a requirements.txt file</li>
            </ul>
        '''
    }
}

# Dictionary of guided tours
GUIDED_TOURS = {
    'document_workflow': {
        'title': 'Document Management Workflow',
        'steps': [
            {
                'id': 'step1',
                'title': 'Welcome to Document Management',
                'content': 'This tour will guide you through the document management workflow.',
                'element': '#document-management-header',
                'position': 'bottom',
                'isFirst': True,
                'isLast': False
            },
            {
                'id': 'step2',
                'title': 'Upload Documents',
                'content': 'Click here to upload new documents to your collection.',
                'element': '#upload-button',
                'position': 'bottom',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step3',
                'title': 'Document List',
                'content': 'Here you can see all your uploaded documents.',
                'element': '#document-list',
                'position': 'top',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step4',
                'title': 'Document Actions',
                'content': 'Use these buttons to view or delete documents.',
                'element': '.document-actions',
                'position': 'left',
                'isFirst': False,
                'isLast': True
            }
        ]
    },
    'vector_db_workflow': {
        'title': 'Vector Database Creation Workflow',
        'steps': [
            {
                'id': 'step1',
                'title': 'Welcome to Vector Database Creation',
                'content': 'This tour will guide you through creating a vector database.',
                'element': '#vector-db-header',
                'position': 'bottom',
                'isFirst': True,
                'isLast': False
            },
            {
                'id': 'step2',
                'title': 'Select Documents',
                'content': 'First, select the documents you want to include in your vector database.',
                'element': '#document-selection',
                'position': 'bottom',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step3',
                'title': 'Choose Document Loader',
                'content': 'Select a document loader appropriate for your file types.',
                'element': '#loader-selection',
                'position': 'right',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step4',
                'title': 'Configure Text Splitting',
                'content': 'Configure how your documents will be split into chunks.',
                'element': '#text-splitting',
                'position': 'left',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step5',
                'title': 'Select Embedding Model',
                'content': 'Choose an embedding model to convert text to vectors.',
                'element': '#embedding-model',
                'position': 'right',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step6',
                'title': 'Choose Vector Store',
                'content': 'Select a vector store to save your embeddings.',
                'element': '#vector-store',
                'position': 'left',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step7',
                'title': 'Create Vector Database',
                'content': 'Click here to create your vector database.',
                'element': '#create-button',
                'position': 'top',
                'isFirst': False,
                'isLast': True
            }
        ]
    },
    'query_workflow': {
        'title': 'Query Workflow',
        'steps': [
            {
                'id': 'step1',
                'title': 'Welcome to the Query Interface',
                'content': 'This tour will guide you through querying your RAG system.',
                'element': '#query-header',
                'position': 'bottom',
                'isFirst': True,
                'isLast': False
            },
            {
                'id': 'step2',
                'title': 'Select Vector Database',
                'content': 'First, select the vector database you want to query.',
                'element': '#vector-db-select',
                'position': 'bottom',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step3',
                'title': 'Configure Retriever',
                'content': 'Configure how documents are retrieved from the vector database.',
                'element': '#retriever-config',
                'position': 'right',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step4',
                'title': 'Select Language Model',
                'content': 'Choose a language model to generate answers.',
                'element': '#llm-select',
                'position': 'left',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step5',
                'title': 'Enter Query',
                'content': 'Type your question here.',
                'element': '#query-input',
                'position': 'top',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step6',
                'title': 'Submit Query',
                'content': 'Click here to submit your question and get an answer.',
                'element': '#submit-button',
                'position': 'right',
                'isFirst': False,
                'isLast': False
            },
            {
                'id': 'step7',
                'title': 'View Results',
                'content': 'The answer and source chunks will appear here.',
                'element': '#results-container',
                'position': 'top',
                'isFirst': False,
                'isLast': True
            }
        ]
    }
}

@bp.route('/')
def index():
    """Render the help index page"""
    return render_template('help/index.html')

@bp.route('/<help_id>')
def get_help(help_id):
    """Get help content for a specific topic"""
    if help_id in HELP_CONTENT:
        return jsonify(HELP_CONTENT[help_id])
    return jsonify({'title': 'Help Not Found', 'content': 'The requested help topic was not found.'}), 404

@bp.route('/error/<error_code>')
def get_error_explanation(error_code):
    """Get explanation for a specific error code"""
    if error_code in ERROR_EXPLANATIONS:
        return jsonify(ERROR_EXPLANATIONS[error_code])
    return jsonify({
        'title': 'Unknown Error',
        'explanation': 'No specific information is available for this error.',
        'solution': 'Try restarting the application or check the logs for more details.'
    })

@bp.route('/tour/<tour_id>')
def get_tour(tour_id):
    """Get guided tour data for a specific workflow"""
    if tour_id in GUIDED_TOURS:
        return jsonify(GUIDED_TOURS[tour_id])
    return jsonify({'error': 'Tour not found'}), 404