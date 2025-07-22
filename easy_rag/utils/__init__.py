"""
Utility modules for Easy RAG System.
"""
from easy_rag.utils.db_utils import (
    document_crud,
    vector_db_crud,
    config_crud,
    query_result_crud
)
from easy_rag.utils.migrations import migration_manager
from easy_rag.utils.db_operations import (
    initialize_database,
    backup_database,
    restore_database,
    export_documents_metadata,
    export_vector_dbs_metadata,
    import_documents_metadata,
    import_vector_dbs_metadata,
    get_database_stats,
    clear_query_history,
    vacuum_database,
    check_database_integrity,
    fix_orphaned_vector_dbs
)

__all__ = [
    'document_crud',
    'vector_db_crud',
    'config_crud',
    'query_result_crud',
    'migration_manager',
    'initialize_database',
    'backup_database',
    'restore_database',
    'export_documents_metadata',
    'export_vector_dbs_metadata',
    'import_documents_metadata',
    'import_vector_dbs_metadata',
    'get_database_stats',
    'clear_query_history',
    'vacuum_database',
    'check_database_integrity',
    'fix_orphaned_vector_dbs'
]