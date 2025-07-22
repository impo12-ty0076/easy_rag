import pytest
import json
from flask import url_for
from easy_rag.models import VectorDatabase, Configuration
from easy_rag import db
from easy_rag.utils.retrievers import get_retriever
from easy_rag.utils.llms import get_llm
from easy_rag.utils.query import process_query

def test_retriever_configuration(client, sample_vector_db):
    """Test retriever configuration."""
    # Get the retriever configuration form
    response = client.get('/retriever/configure')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains retriever options
    assert b'Similarity Search' in response.data
    assert b'MMR' in response.data

def test_get_retriever(app, sample_vector_db, monkeypatch):
    """Test getting a retriever."""
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
    
    with app.app_context():
        # Get a similarity retriever
        config = {
            'type': 'similarity',
            'k': 4
        }
        retriever = get_retriever(sample_vector_db, config)
        
        # Check that the retriever was created correctly
        assert retriever is not None
        
        # Test retrieval
        results = retriever.retrieve('test query')
        
        # Check that the retrieval returned results
        assert len(results) == 2
        assert results[0]['page_content'] == 'This is a test chunk 1.'

def test_llm_selection(client):
    """Test LLM selection."""
    # Get the LLM selection form
    response = client.get('/llm/select')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the form contains LLM options
    assert b'OpenAI' in response.data
    assert b'Hugging Face' in response.data

def test_get_llm(app_context, monkeypatch):
    """Test getting an LLM."""
    # Mock the LLM to avoid actual API calls or model loading
    def mock_get_openai_llm(model_name, api_key, temperature):
        class MockLLM:
            def generate(self, prompt, context=None):
                return "This is a test response."
        
        return MockLLM()
    
    monkeypatch.setattr('easy_rag.utils.llms.get_openai_llm', mock_get_openai_llm)
    
    # Get an OpenAI LLM
    config = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7
    }
    llm = get_llm(config)
    
    # Check that the LLM was created correctly
    assert llm is not None
    
    # Test generation
    response = llm.generate("Test prompt")
    
    # Check that the generation returned a response
    assert response == "This is a test response."

def test_query_processing(app, sample_vector_db, monkeypatch):
    """Test query processing."""
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
        # Create a configuration
        config = Configuration(
            name='test_config',
            settings={
                'retriever': {
                    'type': 'similarity',
                    'k': 4
                },
                'llm': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.7
                }
            }
        )
        db.session.add(config)
        db.session.commit()
        
        # Process a query
        result = process_query(
            query="What is the test about?",
            vector_db_id=sample_vector_db.id,
            config_id=config.id
        )
        
        # Check that the query processing returned a result
        assert result is not None
        assert result['response'] == "This is a test response."
        assert len(result['contexts']) == 2
        assert result['contexts'][0]['content'] == 'This is a test chunk 1.'