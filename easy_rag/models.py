from datetime import datetime
from easy_rag import db

class Document(db.Model):
    """Model for document metadata"""
    id = db.Column(db.String(36), primary_key=True)
    path = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    size = db.Column(db.Integer, nullable=False)
    last_modified = db.Column(db.DateTime, nullable=False, default=datetime.now)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    doc_metadata = db.Column(db.JSON, nullable=True)
    
    def to_dict(self):
        """Convert document to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'path': self.path,
            'name': self.name,
            'type': self.type,
            'size': self.size,
            'last_modified': self.last_modified.strftime('%Y-%m-%d %H:%M:%S'),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'doc_metadata': self.doc_metadata
        }
    
    def __json__(self):
        """JSON serialization method"""
        return self.to_dict()
        
    @property
    def document_metadata(self):
        """Alias for doc_metadata to maintain backward compatibility"""
        return self.doc_metadata

class VectorDatabase(db.Model):
    """Model for vector database metadata"""
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    document_ids = db.Column(db.JSON, nullable=False)  # List of document IDs
    embedding_model = db.Column(db.String(255), nullable=False)
    vector_store_type = db.Column(db.String(50), nullable=False)
    text_splitter = db.Column(db.JSON, nullable=False)  # Text splitter configuration
    chunk_count = db.Column(db.Integer, nullable=False)
    db_metadata = db.Column(db.JSON, nullable=True)

class Configuration(db.Model):
    """Model for system configuration"""
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    settings = db.Column(db.JSON, nullable=False)

class QueryResult(db.Model):
    """Model for query results"""
    id = db.Column(db.String(36), primary_key=True)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    contexts = db.Column(db.JSON, nullable=False)  # Retrieved contexts
    llm_used = db.Column(db.String(255), nullable=False)
    retriever_used = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)