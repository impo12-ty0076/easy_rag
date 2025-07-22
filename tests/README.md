# Easy RAG System Tests

This directory contains tests for the Easy RAG System. The tests are organized into unit tests and integration tests.

## Test Structure

- `conftest.py`: Contains fixtures for setting up test environments
- `test_models.py`: Tests for database models
- `test_document_management.py`: Tests for document management functionality
- `test_vector_db_creation.py`: Tests for vector database creation
- `test_query_processing.py`: Tests for query processing
- `test_diagnostics.py`: Tests for system diagnostics
- `test_integration.py`: Integration tests for end-to-end workflows

## Running Tests

To run the tests, use the following command from the project root directory:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_models.py
```

To run tests with verbose output:

```bash
pytest -v
```

To run tests with coverage report:

```bash
pytest --cov=easy_rag
```

## Test Environment

The tests use a temporary SQLite database and temporary directories for uploads and vector databases. This ensures that the tests do not interfere with the actual application data.

## Mocking

Some tests use mocking to avoid actual API calls, model loading, and disk operations. This makes the tests faster and more reliable.

## Integration Tests

The integration tests simulate end-to-end workflows, including:

1. Document upload and vector database creation
2. Retriever configuration and query processing
3. Error handling
4. Performance testing

## Adding New Tests

When adding new functionality to the Easy RAG System, make sure to add corresponding tests. Follow these guidelines:

1. Add unit tests for new components
2. Update integration tests if the workflow changes
3. Use fixtures from `conftest.py` when possible
4. Mock external dependencies to avoid actual API calls and model loading