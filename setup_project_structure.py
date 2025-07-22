"""
Setup script for Easy RAG System project structure.
This script creates the necessary directories for the Flask application.
"""
import os
import sys

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def setup_project_structure():
    """Set up the project directory structure."""
    # Define the directories to create
    directories = [
        'easy_rag',
        'easy_rag/routes',
        'easy_rag/static',
        'easy_rag/static/css',
        'easy_rag/static/js',
        'easy_rag/static/img',
        'easy_rag/templates',
        'easy_rag/templates/document',
        'easy_rag/templates/vector_db',
        'easy_rag/templates/retriever',
        'easy_rag/templates/query',
        'easy_rag/utils',
        'easy_rag/utils/db',
        'instance',
        'instance/uploads',
        'instance/vector_dbs',
    ]
    
    # Create the directories
    for directory in directories:
        create_directory(directory)
    
    print("\nProject structure setup completed successfully!")

if __name__ == '__main__':
    setup_project_structure()