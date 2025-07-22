"""
Tests for the help system functionality
"""
import pytest
from flask import url_for
import json

def test_help_index_page(client):
    """Test that the help index page loads correctly"""
    response = client.get('/help/')
    assert response.status_code == 200
    assert b'Welcome to Easy RAG System Help' in response.data
    assert b'Getting Started' in response.data
    assert b'Guided Tours' in response.data

def test_help_content_endpoint(client):
    """Test that the help content endpoint returns the correct data"""
    response = client.get('/help/document_management')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'title' in data
    assert 'content' in data
    assert data['title'] == 'Document Management'
    assert 'Managing Your Documents' in data['content']

def test_error_explanation_endpoint(client):
    """Test that the error explanation endpoint returns the correct data"""
    response = client.get('/help/error/db_connection')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'title' in data
    assert 'explanation' in data
    assert 'solution' in data
    assert data['title'] == 'Database Connection Error'

def test_guided_tour_endpoint(client):
    """Test that the guided tour endpoint returns the correct data"""
    response = client.get('/help/tour/document_workflow')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'title' in data
    assert 'steps' in data
    assert data['title'] == 'Document Management Workflow'
    assert len(data['steps']) > 0
    assert 'id' in data['steps'][0]
    assert 'title' in data['steps'][0]
    assert 'content' in data['steps'][0]
    assert 'element' in data['steps'][0]

def test_nonexistent_help_content(client):
    """Test that requesting nonexistent help content returns a 404 status code"""
    response = client.get('/help/nonexistent_topic')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'title' in data
    assert data['title'] == 'Help Not Found'

def test_nonexistent_guided_tour(client):
    """Test that requesting a nonexistent guided tour returns a 404 status code"""
    response = client.get('/help/tour/nonexistent_tour')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Tour not found'

def test_unknown_error_code(client):
    """Test that requesting an unknown error code returns a generic explanation"""
    response = client.get('/help/error/unknown_error')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'title' in data
    assert 'explanation' in data
    assert 'solution' in data
    assert data['title'] == 'Unknown Error'