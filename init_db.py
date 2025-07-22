"""
Database initialization script for Easy RAG System.
This script initializes the SQLite database with the required schema.
"""
import os
import sys
from flask import Flask
from easy_rag import create_app
from easy_rag.utils import initialize_database
from easy_rag.utils.migrations import migration_manager
from easy_rag.errors import DatabaseError, MigrationError

def init_database():
    """Initialize the database with schema and default configuration."""
    print("Initializing database...")
    
    # Create a Flask application context
    app = create_app()
    
    with app.app_context():
        try:
            # Initialize database schema and default configuration
            initialize_database()
            print("Database tables created and initialized.")
            
            # Manually create migrations table
            try:
                migration_manager._create_migrations_table()
                print("Migrations table created.")
            except Exception as e:
                print(f"Error creating migrations table: {str(e)}")
            
            # Apply any pending migrations
            try:
                pending_migrations = migration_manager.get_pending_migrations()
                if pending_migrations:
                    print(f"Applying {len(pending_migrations)} pending migrations...")
                    applied = migration_manager.migrate()
                    print(f"Applied {len(applied)} migrations successfully.")
                else:
                    print("No pending migrations to apply.")
            except Exception as e:
                print(f"Error with migrations: {str(e)}")
            
            print("Database initialization completed successfully!")
            return True
        except (DatabaseError, MigrationError) as e:
            print(f"Error initializing database: {str(e)}")
            return False

if __name__ == '__main__':
    success = init_database()
    sys.exit(0 if success else 1)