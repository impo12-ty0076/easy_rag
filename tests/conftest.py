import os
import tempfile
import pytest
from easy_rag import create_app, db
from easy_rag.models import Document, VectorDatabase, Configuration

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Create a temporary file to isolate the database for each test
    db_fd, db_path = tempfile.mkstemp()
    # Create a temporary directory for uploads and vector databases
    temp_dir = tempfile.mkdtemp()
    upload_folder = os.path.join(temp_dir, 'uploads')
    vector_db_folder = os.path.join(temp_dir, 'vector_dbs')
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(vector_db_folder, exist_ok=True)
    
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'UPLOAD_FOLDER': upload_folder,
        'VECTOR_DB_FOLDER': vector_db_folder,
        'WTF_CSRF_ENABLED': False,  # Disable CSRF protection in tests
    })
    
    # Create the database and the tables
    with app.app_context():
        db.create_all()
    
    yield app
    
    # Close and remove the temporary database
    os.close(db_fd)
    os.unlink(db_path)
    
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test CLI runner for the app."""
    return app.test_cli_runner()

@pytest.fixture
def app_context(app):
    """An application context for the app."""
    with app.app_context():
        yield

@pytest.fixture
def sample_document(app):
    """Create a sample document for testing."""
    with app.app_context():
        doc = Document(
            name='test_document.txt',
            path='/path/to/test_document.txt',
            type='text/plain',
            size=1024,
            content='This is a test document content.'
        )
        db.session.add(doc)
        db.session.commit()
        return doc

@pytest.fixture
def sample_vector_db(app, sample_document):
    """Create a sample vector database for testing."""
    with app.app_context():
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
        return vector_db

@pytest.fixture
def sample_configuration(app):
    """Create a sample configuration for testing."""
    with app.app_context():
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
        return config