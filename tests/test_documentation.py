"""
Tests for the documentation system
"""
import pytest
from flask import url_for

def test_docs_index_page(client):
    """Test that the documentation index page loads correctly"""
    response = client.get('/docs/')
    assert response.status_code == 200
    assert b'Easy RAG System Documentation' in response.data
    assert b'Setup Guide' in response.data
    assert b'Usage Tutorials' in response.data
    assert b'Troubleshooting' in response.data

def test_docs_setup_guide(client):
    """Test that the setup guide page loads correctly"""
    response = client.get('/docs/setup_guide')
    assert response.status_code == 200
    assert b'Setup Guide' in response.data
    assert b'Installation Steps' in response.data

def test_docs_usage_tutorials(client):
    """Test that the usage tutorials page loads correctly"""
    response = client.get('/docs/usage_tutorials')
    assert response.status_code == 200
    assert b'Usage Tutorials' in response.data
    assert b'Managing Documents' in response.data

def test_docs_troubleshooting(client):
    """Test that the troubleshooting page loads correctly"""
    response = client.get('/docs/troubleshooting')
    assert response.status_code == 200
    assert b'Troubleshooting Guide' in response.data
    assert b'Installation Issues' in response.data

def test_nonexistent_doc_page(client):
    """Test that requesting a nonexistent documentation page returns a 404 status code"""
    response = client.get('/docs/nonexistent_page')
    assert response.status_code == 404