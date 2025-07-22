import os
import pytest
import io
from flask import url_for
from easy_rag.models import Document
from easy_rag import db
from easy_rag.utils.document_loaders import get_document_content

def test_document_upload(client, app):
    """Test document upload functionality."""
    # Create a test file
    data = {
        'file': (io.BytesIO(b'This is a test document content.'), 'test_document.txt')
    }
    
    # Upload the file
    response = client.post('/document/upload', data=data, content_type='multipart/form-data')
    
    # Check that the response is a redirect
    assert response.status_code == 302
    
    # Check that the document was created in the database
    with app.app_context():
        doc = Document.query.filter_by(name='test_document.txt').first()
        assert doc is not None
        assert doc.type == 'text/plain'
        assert doc.size > 0
        
        # Check that the file was saved to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], doc.path)
        assert os.path.exists(file_path)

def test_document_list(client, sample_document):
    """Test document listing functionality."""
    # Get the document list
    response = client.get('/document/list')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the document is in the response
    assert b'test_document.txt' in response.data

def test_document_delete(client, app, sample_document):
    """Test document deletion functionality."""
    # Delete the document
    response = client.post(f'/document/delete/{sample_document.id}')
    
    # Check that the response is a redirect
    assert response.status_code == 302
    
    # Check that the document was deleted from the database
    with app.app_context():
        doc = Document.query.get(sample_document.id)
        assert doc is None

def test_document_content_extraction(app, sample_document):
    """Test document content extraction."""
    with app.app_context():
        # Create a test file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_content.txt')
        with open(file_path, 'w') as f:
            f.write('This is a test document content.')
        
        # Update the document path
        sample_document.path = 'test_content.txt'
        db.session.commit()
        
        # Extract content
        content = get_document_content(sample_document)
        
        # Check that the content was extracted correctly
        assert content == 'This is a test document content.'
        
        # Clean up
        os.remove(file_path)