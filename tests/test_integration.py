import pytest
import os
import io
import time
import json
from flask import url_for
from easy_rag.models import Document, VectorDatabase, Configuration
from easy_rag import db

class TestEndToEndWorkflow:
    """Test the end-to-end workflow of the Easy RAG System."""
    
    def test_document_upload_and_vector_db_creation(self, client, app, monkeypatch):
        """Test uploading a document and creating a vector database."""
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
        
        # Step 1: Upload a document
        data = {
            'file': (io.BytesIO(b'This is a test document content.'), 'test_document.txt')
        }
        
        response = client.post('/document/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 302  # Redirect after successful upload
        
        # Check that the document was created in the database
        with app.app_context():
            doc = Document.query.filter_by(name='test_document.txt').first()
            assert doc is not None
            
            # Step 2: Start vector database creation
            response = client.get('/vector_db/create')
            assert response.status_code == 200
            
            # Step 3: Select the document
            response = client.post('/vector_db/select_documents', data={
                'document_ids': [doc.id]
            })
            assert response.status_code == 302  # Redirect to next step
            
            # Step 4: Configure text splitter
            response = client.post('/vector_db/text_splitter', data={
                'splitter_type': 'character',
                'chunk_size': '1000',
                'chunk_overlap': '200'
            })
            assert response.status_code == 302  # Redirect to next step
            
            # Step 5: Select embedding model
            response = client.post('/vector_db/embedding_model', data={
                'embedding_model': 'all-MiniLM-L6-v2'
            })
            assert response.status_code == 302  # Redirect to next step
            
            # Step 6: Select vector store
            response = client.post('/vector_db/vector_store', data={
                'vector_store_type': 'chroma'
            })
            assert response.status_code == 302  # Redirect to next step
            
            # Step 7: Create the vector database
            response = client.post('/vector_db/create_process', data={
                'name': 'test_vector_db'
            })
            assert response.status_code == 302  # Redirect after successful creation
            
            # Check that the vector database was created
            vector_db = VectorDatabase.query.filter_by(name='test_vector_db').first()
            assert vector_db is not None
            assert vector_db.embedding_model == 'all-MiniLM-L6-v2'
            assert vector_db.vector_store_type == 'chroma'
            assert vector_db.document_ids == [doc.id]
    
    def test_retriever_configuration_and_query(self, client, app, sample_vector_db, monkeypatch):
        """Test configuring a retriever and running a query."""
        # Mock the vector store loading to avoid actual disk operations
        def mock_load_vector_store(vector_db):
            class MockVectorStore:
                def similarity_search(self, query, k=4):
                    return [
                        {'page_content': 'This is a test chunk 1.', 'metadata': {'source': 'test_document.txt'}},
                        {'page_content': 'This is a test chunk 2.', 'metadata': {'source': 'test_document.txt'}}
                    ]
                
                def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5):
                    return [
                        {'page_content': 'This is a test chunk 1.', 'metadata': {'source': 'test_document.txt'}},
                        {'page_content': 'This is a test chunk 2.', 'metadata': {'source': 'test_document.txt'}}
                    ]
            
            return MockVectorStore()
        
        monkeypatch.setattr('easy_rag.utils.vector_stores.load_vector_store', mock_load_vector_store)
        
        # Mock the LLM to avoid actual API calls or model loading
        def mock_get_llm(config):
            class MockLLM:
                def generate(self, prompt, context=None):
                    return "This is a test response."
            
            return MockLLM()
        
        monkeypatch.setattr('easy_rag.utils.llms.get_llm', mock_get_llm)
        
        with app.app_context():
            # Step 1: Configure retriever
            response = client.post('/retriever/configure', data={
                'vector_db_id': sample_vector_db.id,
                'retriever_type': 'similarity',
                'k': '4'
            })
            assert response.status_code == 302  # Redirect after successful configuration
            
            # Step 2: Select LLM
            response = client.post('/llm/select', data={
                'model': 'gpt-3.5-turbo',
                'temperature': '0.7'
            })
            assert response.status_code == 302  # Redirect after successful selection
            
            # Step 3: Save configuration
            response = client.post('/query/save_config', data={
                'name': 'test_config'
            })
            assert response.status_code == 302  # Redirect after successful save
            
            # Check that the configuration was created
            config = Configuration.query.filter_by(name='test_config').first()
            assert config is not None
            assert config.settings['retriever']['type'] == 'similarity'
            assert config.settings['llm']['model'] == 'gpt-3.5-turbo'
            
            # Step 4: Run a query
            response = client.post('/query/process', data={
                'query': 'What is this about?',
                'vector_db_id': sample_vector_db.id,
                'config_id': config.id
            })
            assert response.status_code == 200  # Successful query
            
            # Check that the response contains the expected data
            data = json.loads(response.data)
            assert 'response' in data
            assert 'contexts' in data
            assert data['response'] == 'This is a test response.'
            assert len(data['contexts']) == 2
            assert data['contexts'][0]['content'] == 'This is a test chunk 1.'

class TestErrorHandling:
    """Test error handling in the Easy RAG System."""
    
    def test_invalid_document_upload(self, client):
        """Test uploading an invalid document."""
        # Try to upload an empty file
        data = {
            'file': (io.BytesIO(b''), 'empty.txt')
        }
        
        response = client.post('/document/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 302  # Redirect after error
        
        # Follow the redirect
        response = client.get('/')
        assert b'Error' in response.data or b'error' in response.data
    
    def test_missing_api_key(self, client, app, sample_vector_db, monkeypatch):
        """Test handling missing API keys."""
        # Mock the check_api_key function to simulate missing API key
        def mock_check_api_key(model_name):
            return False
        
        monkeypatch.setattr('easy_rag.utils.embedding_models.check_api_key', mock_check_api_key)
        
        # Try to select an API-based embedding model
        response = client.post('/vector_db/embedding_model', data={
            'embedding_model': 'text-embedding-ada-002'
        })
        
        # Should still redirect but with a flash message
        assert response.status_code == 302
        
        # Follow the redirect
        response = client.get('/vector_db/embedding_model')
        assert b'API key' in response.data
    
    def test_invalid_configuration(self, client):
        """Test handling invalid configuration."""
        # Try to configure a retriever with invalid parameters
        response = client.post('/retriever/configure', data={
            'retriever_type': 'similarity',
            'k': 'not_a_number'  # Invalid k value
        })
        
        # Should still redirect but with a flash message
        assert response.status_code == 302
        
        # Follow the redirect
        response = client.get('/retriever/configure')
        assert b'Error' in response.data or b'error' in response.data or b'Invalid' in response.data

class TestPerformance:
    """Test performance of the Easy RAG System."""
    
    def test_document_upload_performance(self, client, app):
        """Test document upload performance."""
        # Create a large test file (1MB)
        large_content = b'x' * (1024 * 1024)  # 1MB of data
        
        # Measure upload time
        start_time = time.time()
        
        data = {
            'file': (io.BytesIO(large_content), 'large_file.txt')
        }
        
        response = client.post('/document/upload', data=data, content_type='multipart/form-data')
        
        end_time = time.time()
        upload_time = end_time - start_time
        
        # Check that the upload was successful
        assert response.status_code == 302
        
        # Check that the upload time is reasonable (less than 5 seconds)
        assert upload_time < 5.0, f"Upload took too long: {upload_time} seconds"
        
        # Clean up
        with app.app_context():
            doc = Document.query.filter_by(name='large_file.txt').first()
            if doc:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc.path)
                if os.path.exists(file_path):
                    os.remove(file_path)
                db.session.delete(doc)
                db.session.commit()
    
    def test_query_performance(self, client, app, sample_vector_db, sample_configuration, monkeypatch):
        """Test query processing performance."""
        # Mock the vector store loading to avoid actual disk operations
        def mock_load_vector_store(vector_db):
            class MockVectorStore:
                def similarity_search(self, query, k=4):
                    return [
                        {'page_content': 'This is a test chunk 1.', 'metadata': {'source': 'test_document.txt'}},
                        {'page_content': 'This is a test chunk 2.', 'metadata': {'source': 'test_document.txt'}}
                    ]
            
            return MockVectorStore()
        
        monkeypatch.setattr('easy_rag.utils.vector_stores.load_vector_store', mock_load_vector_store)
        
        # Mock the LLM to avoid actual API calls or model loading
        def mock_get_llm(config):
            class MockLLM:
                def generate(self, prompt, context=None):
                    return "This is a test response."
            
            return MockLLM()
        
        monkeypatch.setattr('easy_rag.utils.llms.get_llm', mock_get_llm)
        
        # Measure query processing time
        start_time = time.time()
        
        response = client.post('/query/process', data={
            'query': 'What is this about?',
            'vector_db_id': sample_vector_db.id,
            'config_id': sample_configuration.id
        })
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Check that the query was successful
        assert response.status_code == 200
        
        # Check that the query time is reasonable (less than 2 seconds)
        assert query_time < 2.0, f"Query processing took too long: {query_time} seconds"