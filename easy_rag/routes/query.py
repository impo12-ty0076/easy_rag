from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for
from easy_rag import db
from easy_rag.models import VectorDatabase, QueryResult, Configuration
from easy_rag.utils.query import process_query as process_query_util
import uuid
from datetime import datetime

bp = Blueprint('query', __name__, url_prefix='/query')

@bp.route('/', methods=['GET'])
def index():
    """Query interface main page"""
    # Get available vector databases
    vector_dbs = VectorDatabase.query.all()
    
    # Get recent queries
    recent_queries = QueryResult.query.order_by(QueryResult.timestamp.desc()).limit(5).all()
    
    # Get current LLM configuration if it exists
    config = Configuration.query.filter_by(name="llm_config").first()
    current_config = config.settings if config else {}
    
    return render_template('query/index.html', 
                          vector_dbs=vector_dbs,
                          recent_queries=recent_queries,
                          current_config=current_config)

@bp.route('/process', methods=['POST'])
def process_query():
    """Process a query using the selected vector database and LLM"""
    query_text = request.form.get('query', '')
    vector_db_id = request.form.get('vector_db_id', '')
    llm_id = request.form.get('llm_id', '')
    retriever_type = request.form.get('retriever_type', 'similarity')  # Default to similarity search
    
    # Validate required parameters
    if not query_text:
        return jsonify({"error": "Query text is required", "success": False}), 400
    
    if not vector_db_id:
        return jsonify({"error": "Vector database selection is required", "success": False}), 400
    
    if not llm_id:
        return jsonify({"error": "Language model selection is required", "success": False}), 400
    
    try:
        # Process the query using our utility function
        result = process_query_util(query_text, vector_db_id, llm_id, retriever_type)
        
        if not result.get("success", False):
            # If there was an error in processing, return it
            return jsonify({"error": result.get("error", "Unknown error"), "success": False}), 400
        
        # Save the query result to the database
        query_result = QueryResult(
            id=str(uuid.uuid4()),
            query=query_text,
            response=result["response"],
            contexts=result["contexts"],
            llm_used=result["llm_used"],
            retriever_used=result["retriever_used"],
            timestamp=datetime.now()
        )
        
        db.session.add(query_result)
        db.session.commit()
        
        # Return the result
        return jsonify({
            "query": query_text,
            "response": result["response"],
            "contexts": result["contexts"],
            "llm_used": result["llm_used"],
            "retriever_used": result["retriever_used"],
            "success": True
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}", "success": False}), 500

@bp.route('/history', methods=['GET'])
def query_history():
    """Display query history"""
    queries = QueryResult.query.order_by(QueryResult.timestamp.desc()).all()
    return render_template('query/history.html', queries=queries)

@bp.route('/history/<query_id>', methods=['GET'])
def query_detail(query_id):
    """Display details of a specific query"""
    query = QueryResult.query.get_or_404(query_id)
    return render_template('query/detail.html', query=query)