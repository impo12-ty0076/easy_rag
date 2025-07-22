"""
Documentation routes for Easy RAG System

This module provides routes for serving the user documentation.
"""
from flask import Blueprint, render_template, current_app, send_from_directory, abort
import os
import markdown
import markdown.extensions.fenced_code
import markdown.extensions.tables
import markdown.extensions.toc

bp = Blueprint('docs', __name__, url_prefix='/docs')

@bp.route('/')
def index():
    """Render the documentation index page"""
    return render_template('docs/index.html', active_page='index')

@bp.route('/<path:filename>')
def document(filename):
    """Render a specific documentation page"""
    # Check if the requested file exists
    docs_dir = os.path.join(current_app.static_folder, 'docs')
    file_path = os.path.join(docs_dir, f"{filename}.md")
    
    if not os.path.isfile(file_path):
        abort(404)
    
    # Read the markdown file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'fenced_code',
        'tables',
        'toc',
        'codehilite',
        'attr_list'
    ])
    html_content = md.convert(content)
    
    # Extract title from the first heading
    title = filename.replace('_', ' ').title()
    if md.toc_tokens and len(md.toc_tokens) > 0:
        title = md.toc_tokens[0]['name']
    
    return render_template('docs/document.html', 
                          title=title,
                          content=html_content,
                          toc=md.toc,
                          active_page=filename)

@bp.route('/assets/<path:filename>')
def assets(filename):
    """Serve documentation assets"""
    return send_from_directory(os.path.join(current_app.static_folder, 'docs', 'assets'), filename)