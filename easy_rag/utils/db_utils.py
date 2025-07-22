"""
Database utility functions for Easy RAG System.
This module provides CRUD operations and other database utilities.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, TypeVar, Generic, Type
from sqlalchemy.exc import SQLAlchemyError
from easy_rag import db
from easy_rag.models import Document, VectorDatabase, Configuration, QueryResult
from easy_rag.errors import DatabaseError

# Generic type for database models
T = TypeVar('T')

class CRUDBase(Generic[T]):
    """Base class for CRUD operations on database models."""
    
    def __init__(self, model: Type[T]):
        """Initialize with the model class."""
        self.model = model
    
    def create(self, **kwargs) -> T:
        """Create a new record."""
        try:
            # Generate UUID if not provided
            if 'id' not in kwargs:
                kwargs['id'] = str(uuid.uuid4())
            
            # Set timestamps if applicable
            if hasattr(self.model, 'created_at') and 'created_at' not in kwargs:
                kwargs['created_at'] = datetime.now()
            if hasattr(self.model, 'updated_at') and 'updated_at' not in kwargs:
                kwargs['updated_at'] = datetime.now()
            
            obj = self.model(**kwargs)
            db.session.add(obj)
            db.session.commit()
            return obj
        except SQLAlchemyError as e:
            db.session.rollback()
            raise DatabaseError(f"Error creating {self.model.__name__}: {str(e)}")
    
    def get_by_id(self, id: str) -> Optional[T]:
        """Get a record by ID."""
        try:
            return self.model.query.get(id)
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving {self.model.__name__}: {str(e)}")
    
    def get_all(self) -> List[T]:
        """Get all records."""
        try:
            return self.model.query.all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving all {self.model.__name__}s: {str(e)}")
    
    def update(self, id: str, **kwargs) -> Optional[T]:
        """Update a record by ID."""
        try:
            obj = self.get_by_id(id)
            if not obj:
                return None
            
            # Update timestamp if applicable
            if hasattr(obj, 'updated_at'):
                kwargs['updated_at'] = datetime.now()
            
            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            
            db.session.commit()
            return obj
        except SQLAlchemyError as e:
            db.session.rollback()
            raise DatabaseError(f"Error updating {self.model.__name__}: {str(e)}")
    
    def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        try:
            obj = self.get_by_id(id)
            if not obj:
                return False
            
            db.session.delete(obj)
            db.session.commit()
            return True
        except SQLAlchemyError as e:
            db.session.rollback()
            raise DatabaseError(f"Error deleting {self.model.__name__}: {str(e)}")
    
    def filter_by(self, **kwargs) -> List[T]:
        """Filter records by attributes."""
        try:
            return self.model.query.filter_by(**kwargs).all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error filtering {self.model.__name__}s: {str(e)}")


# Create CRUD handlers for each model
class DocumentCRUD(CRUDBase[Document]):
    """CRUD operations for Document model."""
    
    def __init__(self):
        super().__init__(Document)
    
    def get_by_path(self, path: str) -> Optional[Document]:
        """Get a document by its file path."""
        try:
            return Document.query.filter_by(path=path).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving document by path: {str(e)}")
    
    def get_by_type(self, file_type: str) -> List[Document]:
        """Get documents by file type."""
        try:
            return Document.query.filter_by(type=file_type).all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving documents by type: {str(e)}")


class VectorDatabaseCRUD(CRUDBase[VectorDatabase]):
    """CRUD operations for VectorDatabase model."""
    
    def __init__(self):
        super().__init__(VectorDatabase)
    
    def get_by_name(self, name: str) -> Optional[VectorDatabase]:
        """Get a vector database by name."""
        try:
            return VectorDatabase.query.filter_by(name=name).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving vector database by name: {str(e)}")
    
    def get_by_embedding_model(self, model_name: str) -> List[VectorDatabase]:
        """Get vector databases by embedding model."""
        try:
            return VectorDatabase.query.filter_by(embedding_model=model_name).all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving vector databases by embedding model: {str(e)}")
    
    def get_by_document_id(self, document_id: str) -> List[VectorDatabase]:
        """Get vector databases containing a specific document."""
        try:
            # This is a more complex query since document_ids is stored as JSON
            # We'll need to filter in Python
            all_dbs = self.get_all()
            return [db for db in all_dbs if document_id in db.document_ids]
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving vector databases by document ID: {str(e)}")


class ConfigurationCRUD(CRUDBase[Configuration]):
    """CRUD operations for Configuration model."""
    
    def __init__(self):
        super().__init__(Configuration)
    
    def get_by_name(self, name: str) -> Optional[Configuration]:
        """Get a configuration by name."""
        try:
            return Configuration.query.filter_by(name=name).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving configuration by name: {str(e)}")
    
    def get_default_config(self) -> Optional[Configuration]:
        """Get the default configuration."""
        return self.get_by_name('default')
    
    def update_setting(self, config_id: str, key: str, value: Any) -> Optional[Configuration]:
        """Update a specific setting in a configuration."""
        try:
            config = self.get_by_id(config_id)
            if not config:
                return None
            
            # Update the specific setting
            settings = config.settings
            settings[key] = value
            
            # Update the configuration
            return self.update(config_id, settings=settings)
        except SQLAlchemyError as e:
            db.session.rollback()
            raise DatabaseError(f"Error updating configuration setting: {str(e)}")


class QueryResultCRUD(CRUDBase[QueryResult]):
    """CRUD operations for QueryResult model."""
    
    def __init__(self):
        super().__init__(QueryResult)
    
    def get_recent_queries(self, limit: int = 10) -> List[QueryResult]:
        """Get recent query results, ordered by timestamp."""
        try:
            return QueryResult.query.order_by(QueryResult.timestamp.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error retrieving recent queries: {str(e)}")
    
    def search_by_query_text(self, search_text: str) -> List[QueryResult]:
        """Search query results by query text."""
        try:
            return QueryResult.query.filter(QueryResult.query.contains(search_text)).all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Error searching queries: {str(e)}")


# Create singleton instances for easy import
document_crud = DocumentCRUD()
vector_db_crud = VectorDatabaseCRUD()
config_crud = ConfigurationCRUD()
query_result_crud = QueryResultCRUD()