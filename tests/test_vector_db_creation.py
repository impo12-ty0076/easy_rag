import os
import pytest
import json
from flask import url_for
from easy_rag.models import VectorDatabase
from easy_rag import db
from easy_rag.utils.text_splitters import get_text_splitter
from easy_rag.utils.embedding_models import get_embedding_model

def test_vector_db_creation_form(client, sample_document):
    """Test vector database creation form."""
    # Get the vector database creation form
    response = client.get('/vector_db/create')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains the document
    assert b'test_document.txt' in response.data

def test_text_splitter_config(client):
    """Test text splitter configuration."""
    # Get the text splitter configuration form
    response = client.get('/vector_db/text_splitter')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains the text splitter options
    assert b'Character Text Splitter' in response.data
    assert b'Recursive Character Text Splitter' in response.data

def test_get_text_splitter(app_context):
    """Test getting a text splitter."""
    # Get a character text splitter
    config = {
        'type': 'character',
        'chunk_size': 1000,
        'chunk_overlap': 200
    }
    splitter = get_text_splitter(config)
    
    # Check that the splitter was created correctly
    assert splitter is not None
    
    # Test splitting text
    text = "This is a test document content. " * 100
    chunks = splitter.split_text(text)
    
    # Check that the text was split correctly
    assert len(chunks) > 0
    assert len(chunks[0]) <= 1000

def test_embedding_model_selection(client):
    """Test embedding model selection."""
    # Get the embedding model selection form
    response = client.get('/vector_db/embedding_model')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains embedding model options
    assert b'Sentence Transformers' in response.data
    assert b'OpenAI' in response.data

def test_vector_store_selection(client):
    """Test vector store selection."""
    # Get the vector store selection form
    response = client.get('/vector_db/vector_store')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains vector store options
    assert b'Chroma' in response.data
    assert b'FAISS' in response.data
    assert b'Pinecone' in response.data

def test_vector_db_creation_process(client, app, sample_document, monkeypatch):
    """Test the vector database creation process."""
    # Mock the embedding model to avoid actual API calls or model loading
    def mock_get_embedding_model(model_name, api_key=None):
        class MockEmbeddingModel:
            def embed_documents(self, documents):
                return [[0.1] * 384 for _ in documents]
            
            def embed_query(self, query):
                return [0.1] * 384
        
        return MockEmbeddingModel()
    
    monkeypatch.setattr('easy_rag.utils.embedding_models.get_embedding_model', mock_get_embedding_model)
    
    # Mock the vector store creation to avoid actual disk operations
    def mock_create_vector_store(store_type, path):
        class MockVectorStore:
            def add_documents(self, documents, embeddings):
                return True
            
            def save(self):
                return True
        
        return MockVectorStore()
    
    monkeypatch.setattr('easy_rag.utils.vector_stores.create_vector_store', mock_create_vector_store)
    
    # Set up the session with the required data
    with client.session_transaction() as session:
        session['vector_db_creation'] = {
            'document_ids': [sample_document.id],
            'text_splitter_config': {
                'type': 'character',
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_store_type': 'chroma'
        }
    
    # Create a test file for the document
    with app.app_context():
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_content.txt')
        with open(file_path, 'w') as f:
            f.write('This is a test document content.')
        
        # Update the document path
        sample_document.path = 'test_content.txt'
        db.session.commit()
    
    # Submit the vector database creation
    response = client.post('/vector_db/create_process', data={
        'name': 'test_vector_db'
    })
    
    # Check that the response is a redirect
    assert response.status_code == 302
    
    # Check that the vector database was created in the database
    with app.app_context():
        vector_db = VectorDatabase.query.filter_by(name='test_vector_db').first()
        assert vector_db is not None
        assert vector_db.embedding_model == 'all-MiniLM-L6-v2'
        assert vector_db.vector_store_type == 'chroma'
        assert vector_db.document_ids == [sample_document.id]
        
        # Clean up
        os.remove(file_path)