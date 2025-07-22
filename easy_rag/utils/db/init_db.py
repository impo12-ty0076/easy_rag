"""
Database initialization script.
"""
from easy_rag import db
from easy_rag.models import Document, VectorDatabase, Configuration, QueryResult

def init_db():
    """Initialize the database tables."""
    db.create_all()
    
    # Create default configuration if it doesn't exist
    from easy_rag.models import Configuration
    import uuid
    
    default_config = Configuration.query.filter_by(name='default').first()
    if not default_config:
        default_config = Configuration(
            id=str(uuid.uuid4()),
            name='default',
            settings={
                'document_storage_path': 'instance/uploads',
                'vector_db_storage_path': 'instance/vector_dbs',
                'default_chunk_size': 1000,
                'default_chunk_overlap': 200,
                'default_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'default_vector_store': 'chroma',
                'default_retriever': 'similarity',
                'default_llm': 'gpt-3.5-turbo',
            }
        )
        db.session.add(default_config)
        db.session.commit()

if __name__ == '__main__':
    init_db()