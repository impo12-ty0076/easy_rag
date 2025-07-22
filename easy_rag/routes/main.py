from flask import Blueprint, render_template, jsonify
from easy_rag import db
import platform
import sys

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Main entry point of the application"""
    return render_template('index.html')

@bp.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    system_info = {
        'status': 'ok',
        'app_name': 'Easy RAG System',
        'python_version': sys.version,
        'platform': platform.platform(),
        'database_connected': True
    }
    
    # Check database connection
    try:
        db.session.execute('SELECT 1')
        system_info['database_connected'] = True
    except Exception:
        system_info['database_connected'] = False
        system_info['status'] = 'degraded'
    
    return jsonify(system_info)