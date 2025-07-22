"""
Database migration utilities for Easy RAG System.
This module provides tools for managing database schema changes.
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import current_app
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from easy_rag import db
from easy_rag.errors import MigrationError

# Set up logging
logger = logging.getLogger(__name__)

class Migration:
    """Class for handling database migrations."""
    
    def __init__(self, app=None):
        """Initialize the migration manager."""
        self.app = app
        self.migrations_dir = None
        self.migrations_table = 'migrations'
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app."""
        self.app = app
        self.migrations_dir = os.path.join(app.root_path, 'migrations')
        
        # Ensure migrations directory exists
        os.makedirs(self.migrations_dir, exist_ok=True)
        
        # Create migrations table if it doesn't exist
        with app.app_context():
            self._create_migrations_table()
    
    def _create_migrations_table(self):
        """Create the migrations tracking table if it doesn't exist."""
        inspector = inspect(db.engine)
        if self.migrations_table not in inspector.get_table_names():
            try:
                with db.engine.connect() as conn:
                    conn.execute(text(f'''
                        CREATE TABLE {self.migrations_table} (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            applied_at TIMESTAMP NOT NULL,
                            content TEXT NOT NULL
                        )
                    '''))
                    conn.commit()
                logger.info(f"Created migrations table '{self.migrations_table}'")
            except SQLAlchemyError as e:
                logger.error(f"Error creating migrations table: {str(e)}")
                raise MigrationError(f"Failed to create migrations table: {str(e)}")
    
    def create_migration(self, name: str) -> str:
        """Create a new migration file."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        migration_id = f"{timestamp}_{name}"
        filename = f"{migration_id}.json"
        filepath = os.path.join(self.migrations_dir, filename)
        
        # Create migration template
        migration_data = {
            "id": migration_id,
            "name": name,
            "up": [],
            "down": []
        }
        
        # Write migration file
        with open(filepath, 'w') as f:
            json.dump(migration_data, f, indent=2)
        
        logger.info(f"Created migration file: {filename}")
        return filepath
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations."""
        try:
            with db.engine.connect() as conn:
                result = conn.execute(text(f"SELECT id, name, applied_at FROM {self.migrations_table} ORDER BY applied_at"))
                return [{"id": row[0], "name": row[1], "applied_at": row[2]} for row in result]
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving applied migrations: {str(e)}")
            raise MigrationError(f"Failed to retrieve applied migrations: {str(e)}")
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migration files."""
        # Get all migration files
        migration_files = []
        for filename in os.listdir(self.migrations_dir):
            if filename.endswith('.json'):
                migration_files.append(filename.split('.')[0])
        
        # Get applied migrations
        applied_migrations = [m["id"] for m in self.get_applied_migrations()]
        
        # Return pending migrations
        return [m for m in migration_files if m not in applied_migrations]
    
    def apply_migration(self, migration_id: str) -> bool:
        """Apply a specific migration."""
        filepath = os.path.join(self.migrations_dir, f"{migration_id}.json")
        
        if not os.path.exists(filepath):
            logger.error(f"Migration file not found: {migration_id}")
            raise MigrationError(f"Migration file not found: {migration_id}")
        
        # Load migration file
        with open(filepath, 'r') as f:
            migration = json.load(f)
        
        # Check if already applied
        try:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT id FROM {self.migrations_table} WHERE id = :id"),
                    {"id": migration_id}
                ).fetchone()
                
                if result:
                    logger.warning(f"Migration already applied: {migration_id}")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error checking migration status: {str(e)}")
            raise MigrationError(f"Failed to check migration status: {str(e)}")
        
        # Apply migration
        try:
            # Begin transaction
            with db.engine.connect() as conn:
                with conn.begin():
                    # Execute each statement in the 'up' section
                    for statement in migration["up"]:
                        conn.execute(text(statement))
                    
                    # Record migration
                    conn.execute(
                        text(f"INSERT INTO {self.migrations_table} (id, name, applied_at, content) VALUES (:id, :name, :applied_at, :content)"),
                        {
                            "id": migration_id,
                            "name": migration["name"],
                            "applied_at": datetime.now(),
                            "content": json.dumps(migration)
                        }
                    )
            
            logger.info(f"Applied migration: {migration_id}")
            return True
        except Exception as e:
            logger.error(f"Error applying migration {migration_id}: {str(e)}")
            raise MigrationError(f"Failed to apply migration {migration_id}: {str(e)}")
    
    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration."""
        # Check if migration was applied
        try:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT content FROM {self.migrations_table} WHERE id = :id"),
                    {"id": migration_id}
                ).fetchone()
                
                if not result:
                    logger.warning(f"Migration not applied: {migration_id}")
                    return False
                
                # Get migration content
                migration = json.loads(result[0])
        except SQLAlchemyError as e:
            logger.error(f"Error checking migration status: {str(e)}")
            raise MigrationError(f"Failed to check migration status: {str(e)}")
        
        # Rollback migration
        try:
            # Begin transaction
            with db.engine.connect() as conn:
                with conn.begin():
                    # Execute each statement in the 'down' section
                    for statement in migration["down"]:
                        conn.execute(text(statement))
                    
                    # Remove migration record
                    conn.execute(
                        text(f"DELETE FROM {self.migrations_table} WHERE id = :id"),
                        {"id": migration_id}
                    )
            
            logger.info(f"Rolled back migration: {migration_id}")
            return True
        except Exception as e:
            logger.error(f"Error rolling back migration {migration_id}: {str(e)}")
            raise MigrationError(f"Failed to rollback migration {migration_id}: {str(e)}")
    
    def migrate(self) -> List[str]:
        """Apply all pending migrations."""
        pending_migrations = self.get_pending_migrations()
        applied = []
        
        for migration_id in sorted(pending_migrations):
            try:
                if self.apply_migration(migration_id):
                    applied.append(migration_id)
            except MigrationError as e:
                logger.error(f"Migration failed at {migration_id}: {str(e)}")
                break
        
        return applied
    
    def rollback(self, steps: int = 1) -> List[str]:
        """Rollback the last n migrations."""
        applied_migrations = self.get_applied_migrations()
        if not applied_migrations:
            return []
        
        # Get the last n migrations in reverse order
        migrations_to_rollback = applied_migrations[-steps:]
        migrations_to_rollback.reverse()
        
        rolled_back = []
        for migration in migrations_to_rollback:
            try:
                if self.rollback_migration(migration["id"]):
                    rolled_back.append(migration["id"])
            except MigrationError as e:
                logger.error(f"Rollback failed at {migration['id']}: {str(e)}")
                break
        
        return rolled_back


# Create a migration manager instance
migration_manager = Migration()


def init_app(app):
    """Initialize the migration manager with the Flask app."""
    migration_manager.init_app(app)