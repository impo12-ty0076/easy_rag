"""
Database operations utility for Easy RAG System.
This module provides higher-level database operations.
"""
import os
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import current_app
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from easy_rag import db
from easy_rag.models import Document, VectorDatabase, Configuration, QueryResult
from easy_rag.utils.db_utils import document_crud, vector_db_crud, config_crud, query_result_crud
from easy_rag.errors import DatabaseError

def initialize_database():
    """Initialize the database with schema and default configuration."""
    try:
        # Create all tables
        db.create_all()
        
        # Create default configuration if it doesn't exist
        default_config = config_crud.get_by_name('default')
        if not default_config:
            config_crud.create(
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
        
        return True
    except SQLAlchemyError as e:
        raise DatabaseError(f"Error initializing database: {str(e)}")

def backup_database(backup_path: str) -> bool:
    """Backup the database to a file."""
    try:
        # Get database path from app config
        db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Create backup connection
        backup_conn = db.engine.raw_connection()
        
        # Backup database
        with open(backup_path, 'wb') as f:
            for line in backup_conn.iterdump():
                f.write(f'{line}\n'.encode('utf-8'))
        
        backup_conn.close()
        return True
    except Exception as e:
        raise DatabaseError(f"Error backing up database: {str(e)}")

def restore_database(backup_path: str) -> bool:
    """Restore the database from a backup file."""
    try:
        # Get database path from app config
        db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        
        # Check if backup file exists
        if not os.path.exists(backup_path):
            raise DatabaseError(f"Backup file not found: {backup_path}")
        
        # Close all connections
        db.session.close()
        db.engine.dispose()
        
        # Remove existing database file
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # Create new database
        db.create_all()
        
        # Restore from backup
        conn = db.engine.raw_connection()
        with open(backup_path, 'r') as f:
            conn.executescript(f.read())
        
        conn.close()
        return True
    except Exception as e:
        raise DatabaseError(f"Error restoring database: {str(e)}")

def export_documents_metadata(export_path: str) -> bool:
    """Export document metadata to a JSON file."""
    try:
        documents = document_crud.get_all()
        
        # Convert to serializable format
        docs_data = []
        for doc in documents:
            doc_dict = {
                'id': doc.id,
                'path': doc.path,
                'name': doc.name,
                'type': doc.type,
                'size': doc.size,
                'last_modified': doc.last_modified.isoformat(),
                'created_at': doc.created_at.isoformat(),
                'metadata': doc.metadata
            }
            docs_data.append(doc_dict)
        
        # Ensure export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        return True
    except Exception as e:
        raise DatabaseError(f"Error exporting document metadata: {str(e)}")

def export_vector_dbs_metadata(export_path: str) -> bool:
    """Export vector database metadata to a JSON file."""
    try:
        vector_dbs = vector_db_crud.get_all()
        
        # Convert to serializable format
        dbs_data = []
        for db_obj in vector_dbs:
            db_dict = {
                'id': db_obj.id,
                'name': db_obj.name,
                'path': db_obj.path,
                'created_at': db_obj.created_at.isoformat(),
                'document_ids': db_obj.document_ids,
                'embedding_model': db_obj.embedding_model,
                'vector_store_type': db_obj.vector_store_type,
                'text_splitter': db_obj.text_splitter,
                'chunk_count': db_obj.chunk_count,
                'metadata': db_obj.metadata
            }
            dbs_data.append(db_dict)
        
        # Ensure export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(dbs_data, f, indent=2)
        
        return True
    except Exception as e:
        raise DatabaseError(f"Error exporting vector database metadata: {str(e)}")

def import_documents_metadata(import_path: str) -> int:
    """Import document metadata from a JSON file."""
    try:
        # Check if file exists
        if not os.path.exists(import_path):
            raise DatabaseError(f"Import file not found: {import_path}")
        
        # Read file
        with open(import_path, 'r') as f:
            docs_data = json.load(f)
        
        # Import documents
        imported_count = 0
        for doc_dict in docs_data:
            # Check if document already exists
            existing_doc = document_crud.get_by_id(doc_dict['id'])
            if existing_doc:
                continue
            
            # Convert ISO format dates back to datetime
            doc_dict['last_modified'] = datetime.fromisoformat(doc_dict['last_modified'])
            doc_dict['created_at'] = datetime.fromisoformat(doc_dict['created_at'])
            
            # Create document
            document_crud.create(**doc_dict)
            imported_count += 1
        
        return imported_count
    except Exception as e:
        raise DatabaseError(f"Error importing document metadata: {str(e)}")

def import_vector_dbs_metadata(import_path: str) -> int:
    """Import vector database metadata from a JSON file."""
    try:
        # Check if file exists
        if not os.path.exists(import_path):
            raise DatabaseError(f"Import file not found: {import_path}")
        
        # Read file
        with open(import_path, 'r') as f:
            dbs_data = json.load(f)
        
        # Import vector databases
        imported_count = 0
        for db_dict in dbs_data:
            # Check if vector database already exists
            existing_db = vector_db_crud.get_by_id(db_dict['id'])
            if existing_db:
                continue
            
            # Convert ISO format date back to datetime
            db_dict['created_at'] = datetime.fromisoformat(db_dict['created_at'])
            
            # Create vector database
            vector_db_crud.create(**db_dict)
            imported_count += 1
        
        return imported_count
    except Exception as e:
        raise DatabaseError(f"Error importing vector database metadata: {str(e)}")

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics."""
    try:
        # Get counts
        document_count = db.session.query(Document).count()
        vector_db_count = db.session.query(VectorDatabase).count()
        config_count = db.session.query(Configuration).count()
        query_result_count = db.session.query(QueryResult).count()
        
        # Get database size
        db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        
        # Get last query timestamp
        last_query = db.session.query(QueryResult).order_by(QueryResult.timestamp.desc()).first()
        last_query_time = last_query.timestamp if last_query else None
        
        return {
            'document_count': document_count,
            'vector_db_count': vector_db_count,
            'config_count': config_count,
            'query_result_count': query_result_count,
            'database_size_bytes': db_size,
            'last_query_time': last_query_time.isoformat() if last_query_time else None
        }
    except Exception as e:
        raise DatabaseError(f"Error getting database statistics: {str(e)}")

def clear_query_history() -> int:
    """Clear query history and return the number of deleted records."""
    try:
        count = db.session.query(QueryResult).delete()
        db.session.commit()
        return count
    except SQLAlchemyError as e:
        db.session.rollback()
        raise DatabaseError(f"Error clearing query history: {str(e)}")

def vacuum_database() -> bool:
    """Vacuum the SQLite database to optimize storage."""
    try:
        with db.engine.connect() as conn:
            conn.execute(text("VACUUM"))
            conn.commit()
        return True
    except SQLAlchemyError as e:
        raise DatabaseError(f"Error vacuuming database: {str(e)}")

def check_database_integrity() -> Dict[str, Any]:
    """Check database integrity and return results."""
    try:
        # Run integrity check
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA integrity_check")).fetchone()[0]
        
        # Check for orphaned records
        orphaned_vector_dbs = []
        for vdb in vector_db_crud.get_all():
            for doc_id in vdb.document_ids:
                if not document_crud.get_by_id(doc_id):
                    orphaned_vector_dbs.append(vdb.id)
                    break
        
        return {
            'integrity_check': result,
            'orphaned_vector_dbs': orphaned_vector_dbs,
            'status': 'ok' if result == 'ok' and not orphaned_vector_dbs else 'issues_found'
        }
    except Exception as e:
        raise DatabaseError(f"Error checking database integrity: {str(e)}")

def fix_orphaned_vector_dbs() -> int:
    """Fix orphaned vector databases by updating their document_ids."""
    try:
        fixed_count = 0
        for vdb in vector_db_crud.get_all():
            valid_doc_ids = []
            for doc_id in vdb.document_ids:
                if document_crud.get_by_id(doc_id):
                    valid_doc_ids.append(doc_id)
            
            if len(valid_doc_ids) != len(vdb.document_ids):
                vector_db_crud.update(vdb.id, document_ids=valid_doc_ids)
                fixed_count += 1
        
        return fixed_count
    except Exception as e:
        raise DatabaseError(f"Error fixing orphaned vector databases: {str(e)}")