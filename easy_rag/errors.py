"""
Error handling module for Easy RAG System

This module provides comprehensive error handling for the Easy RAG application,
including custom error pages, logging, and user-friendly error messages.
"""
from flask import Blueprint, render_template, current_app, jsonify, request
import traceback
import sys

bp = Blueprint('errors', __name__)

class EasyRAGError(Exception):
    """Base exception class for Easy RAG System"""
    def __init__(self, message, status_code=500, error_code=None, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.payload = payload or {}
        
        # Add error code to payload if provided
        if error_code:
            self.payload['error_code'] = error_code
    
    def to_dict(self):
        """Convert exception to dictionary for API responses"""
        result = dict(self.payload)
        result['message'] = self.message
        result['status'] = 'error'
        if self.error_code:
            result['error_code'] = self.error_code
        return result


class DatabaseError(EasyRAGError):
    """Exception raised for database-related errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='db_connection', payload=payload)


class MigrationError(EasyRAGError):
    """Exception raised for database migration errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='db_connection', payload=payload)


class FileUploadError(EasyRAGError):
    """Exception raised for file upload errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=400, error_code='file_upload', payload=payload)


class VectorDBCreationError(EasyRAGError):
    """Exception raised for vector database creation errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='vector_db_creation', payload=payload)


class DependencyError(EasyRAGError):
    """Exception raised for dependency installation errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='dependency_installation', payload=payload)


class APIKeyError(EasyRAGError):
    """Exception raised for API key errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=401, error_code='api_key', payload=payload)


class ModelDownloadError(EasyRAGError):
    """Exception raised for model download errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='model_download', payload=payload)


class QueryProcessingError(EasyRAGError):
    """Exception raised for query processing errors"""
    def __init__(self, message, payload=None):
        super().__init__(message, status_code=500, error_code='query_processing', payload=payload)

@bp.app_errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Resource not found"}), 404
    return render_template('errors/404.html'), 404

@bp.app_errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    # Log the error with traceback
    exc_info = sys.exc_info()
    current_app.logger.error(f'Server Error: {error}\n{traceback.format_exc()}')
    
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    return render_template('errors/500.html'), 500

@bp.app_errorhandler(403)
def forbidden_error(error):
    """Handle 403 errors"""
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    return render_template('errors/403.html'), 403

@bp.app_errorhandler(400)
def bad_request_error(error):
    """Handle 400 errors"""
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Bad request"}), 400
    return render_template('errors/400.html'), 400

@bp.app_errorhandler(EasyRAGError)
def handle_easy_rag_error(error):
    """Handle custom EasyRAGError exceptions"""
    current_app.logger.error(f'EasyRAGError: {error.message}')
    
    if request.path.startswith('/api/'):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    # For web interface, render appropriate error template
    template = f'errors/{error.status_code}.html'
    try:
        return render_template(template, error=error), error.status_code
    except:
        return render_template('errors/generic.html', error=error), error.status_code

def init_app(app):
    """Register error handlers with the Flask app"""
    app.register_blueprint(bp)
    
    # Setup logging
    if not app.debug:
        import logging
        from logging.handlers import RotatingFileHandler
        import os
        
        # Ensure the logs directory exists
        if not os.path.exists('logs'):
            os.mkdir('logs')
            
        file_handler = RotatingFileHandler('logs/easy_rag.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Easy RAG System startup')
        
    # Register additional error handlers
    app.register_error_handler(Exception, handle_unexpected_error)

def handle_unexpected_error(error):
    """Handle any unhandled exceptions"""
    current_app.logger.error(f'Unhandled Exception: {error}\n{traceback.format_exc()}')
    
    if request.path.startswith('/api/'):
        return jsonify({
            "status": "error", 
            "message": "An unexpected error occurred",
            "error_type": error.__class__.__name__
        }), 500
    
    return render_template('errors/500.html', error=error), 500