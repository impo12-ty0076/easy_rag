# Easy RAG System - Troubleshooting Guide

## Introduction

This troubleshooting guide addresses common issues you might encounter while using the Easy RAG System. For each issue, we provide potential causes and step-by-step solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Document Management Issues](#document-management-issues)
3. [Vector Database Creation Issues](#vector-database-creation-issues)
4. [Retriever Configuration Issues](#retriever-configuration-issues)
5. [Query Processing Issues](#query-processing-issues)
6. [API Key Issues](#api-key-issues)
7. [Performance Issues](#performance-issues)

## Installation Issues

### Python Version Errors

**Issue**: Error messages about incompatible Python version.

**Causes**:
- Python version is too old
- Python version is too new and not fully supported

**Solutions**:
1. Check your Python version: `python --version`
2. Install Python 3.8-3.10 (recommended versions)
3. Create a new virtual environment with the correct Python version
4. Reinstall dependencies in the new environment

### Dependency Installation Failures

**Issue**: Errors when running `pip install -r requirements.txt`.

**Causes**:
- Missing system libraries
- Network issues
- Incompatible package versions

**Solutions**:
1. Install system dependencies:
   - **Windows**: Install Visual C++ Build Tools
   - **Linux**: `sudo apt-get install build-essential python3-dev`
   - **macOS**: `xcode-select --install`
2. Try installing problematic packages individually
3. Check for package conflicts and update requirements.txt if needed
4. Use a different package index: `pip install -r requirements.txt -i https://pypi.org/simple`

### Database Initialization Errors

**Issue**: Errors when running `python init_db.py`.

**Causes**:
- Insufficient permissions
- Corrupted database file
- SQLite version issues

**Solutions**:
1. Run the application with administrator/sudo privileges
2. Delete the existing database file and try again
3. Check SQLite version: `python -c "import sqlite3; print(sqlite3.sqlite_version)"`
4. Ensure SQLite version is 3.24.0 or higher

## Document Management Issues

### File Upload Failures

**Issue**: Unable to upload documents or uploads fail silently.

**Causes**:
- File size too large
- Unsupported file format
- Permission issues with upload directory

**Solutions**:
1. Check file size (limit is 16MB by default)
2. Verify file format is supported
3. Check permissions on the upload directory
4. Increase upload limit in `__init__.py` if needed
5. Check server logs for specific error messages

### Document Content Not Displaying

**Issue**: Document uploads successfully but content doesn't display in preview.

**Causes**:
- Unsupported internal format
- Text extraction failure
- Encoding issues

**Solutions**:
1. Try converting the document to a different format (e.g., PDF to TXT)
2. Check if the document is password-protected or encrypted
3. For PDFs, ensure they contain actual text and not just images
4. Try uploading a smaller portion of the document to isolate the issue

### Document Deletion Errors

**Issue**: Unable to delete documents.

**Causes**:
- File is in use by another process
- Insufficient permissions
- Document is referenced by a vector database

**Solutions**:
1. Close any applications that might be using the file
2. Check file permissions
3. Check if the document is part of any vector databases
4. Restart the application and try again

## Vector Database Creation Issues

### Creation Process Fails

**Issue**: Vector database creation fails or hangs.

**Causes**:
- Document processing errors
- Memory limitations
- Embedding model issues
- Network issues for API-based models

**Solutions**:
1. Check server logs for specific error messages
2. Try with fewer or smaller documents
3. Use a different embedding model
4. For API-based models, check API key and network connectivity
5. Increase system memory or use a machine with more RAM

### Slow Vector Database Creation

**Issue**: Vector database creation takes an extremely long time.

**Causes**:
- Large document collection
- Complex documents
- Slow embedding model
- Limited system resources

**Solutions**:
1. Split the process into smaller batches of documents
2. Use a faster embedding model (local models are often faster)
3. Increase chunk size to reduce the number of embeddings
4. Close other resource-intensive applications
5. Use a more powerful machine if available

### Missing Chunks in Vector Database

**Issue**: Vector database seems incomplete or missing content.

**Causes**:
- Text extraction issues
- Chunk size too large
- Document format issues

**Solutions**:
1. Check text extraction by previewing documents
2. Reduce chunk size to capture more content
3. Try a different text splitting strategy
4. Convert documents to plain text format before uploading

## Retriever Configuration Issues

### Retriever Not Saving Configuration

**Issue**: Retriever configuration doesn't save or apply.

**Causes**:
- Session issues
- Database connection problems
- Invalid configuration parameters

**Solutions**:
1. Refresh the page and try again
2. Check that all required fields are filled
3. Verify parameter values are within acceptable ranges
4. Clear browser cache and cookies
5. Restart the application

### Retriever Performance Issues

**Issue**: Retriever returns irrelevant results.

**Causes**:
- Inappropriate retriever type for the use case
- Poor quality embeddings
- Insufficient number of chunks retrieved
- Poorly formulated queries

**Solutions**:
1. Try a different retriever type
2. Use a higher quality embedding model
3. Increase the number of chunks retrieved
4. Experiment with different retriever parameters
5. Reformulate queries to be more specific

## Query Processing Issues

### No Response from LLM

**Issue**: Query submitted but no response is generated.

**Causes**:
- LLM initialization failure
- API rate limiting
- Network issues
- Insufficient system resources

**Solutions**:
1. Check API key validity for API-based models
2. Verify network connectivity
3. For local models, check system resources (RAM, disk space)
4. Try a smaller or different LLM
5. Check server logs for specific error messages

### Irrelevant or Incorrect Responses

**Issue**: LLM generates responses that are irrelevant or incorrect.

**Causes**:
- Retriever returning irrelevant context
- Insufficient context for the LLM
- Poorly formulated queries
- LLM limitations

**Solutions**:
1. Adjust retriever settings (try different retriever types)
2. Increase the number of chunks retrieved
3. Make queries more specific and clear
4. Try a more capable LLM
5. Check if the information exists in your document collection

### Slow Query Processing

**Issue**: Queries take a long time to process.

**Causes**:
- Large vector database
- Complex retrieval strategy
- Resource-intensive LLM
- System resource limitations

**Solutions**:
1. Use a faster retriever type (Similarity Search is usually fastest)
2. Reduce the number of chunks retrieved
3. Use a smaller or optimized LLM
4. Close other resource-intensive applications
5. Use a more powerful machine if available

## API Key Issues

### API Key Validation Failures

**Issue**: API keys are not accepted or validated.

**Causes**:
- Incorrect API key format
- Expired or invalid API key
- Network issues
- Rate limiting

**Solutions**:
1. Double-check API key for typos
2. Verify API key is active in the provider's dashboard
3. Check network connectivity to the API provider
4. Try a different API key if available
5. Use a local model instead if API issues persist

### API Rate Limiting

**Issue**: Frequent "rate limit exceeded" errors.

**Causes**:
- Too many requests in a short time
- Free tier limitations
- Account billing issues

**Solutions**:
1. Implement request throttling (wait between requests)
2. Upgrade to a paid tier with higher limits
3. Use multiple API keys and rotate between them
4. Cache common queries to reduce API calls
5. Use local models for development and testing

## Performance Issues

### Application Slowness

**Issue**: Overall application performance is slow.

**Causes**:
- Large document collection
- Many vector databases
- Limited system resources
- Browser performance issues

**Solutions**:
1. Archive unused documents and vector databases
2. Close other resource-intensive applications
3. Clear browser cache and cookies
4. Restart the application
5. Use a more powerful machine if available

### Memory Usage Problems

**Issue**: Application uses excessive memory or crashes with out-of-memory errors.

**Causes**:
- Large document processing
- Multiple large vector databases loaded
- Memory leaks
- Resource-intensive LLMs

**Solutions**:
1. Process smaller batches of documents
2. Use more memory-efficient embedding models
3. Close and reopen the application periodically
4. Increase system swap space
5. Add more RAM to your system

### Disk Space Issues

**Issue**: Running out of disk space.

**Causes**:
- Large document collection
- Multiple vector databases
- Downloaded LLMs
- Log files growth

**Solutions**:
1. Clean up unused documents and vector databases
2. Move document and vector database storage to a drive with more space
3. Delete downloaded models that aren't being used
4. Implement log rotation or clear old logs
5. Add more storage to your system

## Getting Additional Help

If you've tried the solutions in this guide and are still experiencing issues:

1. Check the application logs for detailed error messages
2. Search for similar issues in the project's issue tracker
3. Post a detailed description of your issue, including:
   - Error messages
   - Steps to reproduce
   - System specifications
   - Log files
4. Contact support with the above information

## Reporting Bugs

If you believe you've found a bug:

1. Check if it's already been reported
2. Create a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - System information