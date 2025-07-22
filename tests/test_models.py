import pytest
from easy_rag.models import Document, VectorDatabase, Configuration
from easy_rag import db

def test_document_model(app_context):
    """Test the Document model."""
    # Create a new document
    doc = Document(
        name='test_document.txt',
        path='/path/to/test_document.txt',
        type='text/plain',
        size=1024,
        content='This is a test document content.'
    )
    db.session.add(doc)
    db.session.commit()
    
    # Query the document
    retrieved_doc = Document.query.filter_by(name='test_document.txt').first()
    
    # Assert that the document was created correctly
    assert retrieved_doc is not None
    assert retrieved_doc.name == 'test_document.txt'
    assert retrieved_doc.path == '/path/to/test_document.txt'
    assert retrieved_doc.type == 'text/plain'
    assert retrieved_doc.size == 1024
    assert retrieved_doc.content == 'This is a test document content.'

def test_vector_database_model(app_context, sample_document):
    """Test the VectorDatabase model."""
    # Create a new vector database
    vector_db = VectorDatabase(
        name='test_vector_db',
        path='/path/to/test_vector_db',
        embedding_model='test-embedding-model',
        vector_store_type='chroma',
        text_splitter_config={'type': 'character', 'chunk_size': 1000, 'chunk_overlap': 200},
        document_ids=[sample_document.id],
        chunk_count=10
    )
    db.session.add(vector_db)
    db.session.commit()
    
    # Query the vector database
    retrieved_db = VectorDatabase.query.filter_by(name='test_vector_db').first()
    
    # Assert that the vector database was created correctly
    assert retrieved_db is not None
    assert retrieved_db.name == 'test_vector_db'
    assert retrieved_db.path == '/path/to/test_vector_db'
    assert retrieved_db.embedding_model == 'test-embedding-model'
    assert retrieved_db.vector_store_type == 'chroma'
    assert retrieved_db.text_splitter_config == {'type': 'character', 'chunk_size': 1000, 'chunk_overlap': 200}
    assert retrieved_db.document_ids == [sample_document.id]
    assert retrieved_db.chunk_count == 10

def test_configuration_model(app_context):
    """Test the Configuration model."""
    # Create a new configuration
    config = Configuration(
        name='test_config',
        settings={
            'retriever': {
                'type': 'similarity',
                'k': 5
            },
            'llm': {
                'model': 'test-llm-model',
                'temperature': 0.7
            }
        }
    )
    db.session.add(config)
    db.session.commit()
    
    # Query the configuration
    retrieved_config = Configuration.query.filter_by(name='test_config').first()
    
    # Assert that the configuration was created correctly
    assert retrieved_config is not None
    assert retrieved_config.name == 'test_config'
    assert retrieved_config.settings == {
        'retriever': {
            'type': 'similarity',
            'k': 5
        },
        'llm': {
            'model': 'test-llm-model',
            'temperature': 0.7
        }
    }