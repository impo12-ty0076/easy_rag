"""
SQLite database schema for Easy RAG System.
This file defines the database schema using SQLAlchemy models.
"""
from easy_rag import db
from easy_rag.models import Document, VectorDatabase, Configuration, QueryResult

def create_schema():
    """Create the database schema."""
    # Create all tables
    db.create_all()
    
    print("Database schema created successfully.")

def drop_schema():
    """Drop all tables in the database."""
    # Drop all tables
    db.drop_all()
    
    print("Database schema dropped successfully.")

if __name__ == '__main__':
    create_schema()