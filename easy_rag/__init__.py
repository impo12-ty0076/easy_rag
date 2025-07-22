import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app(test_config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__, instance_relative_config=True)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Configure the SQLite database
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(app.instance_path, 'easy_rag.sqlite')}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
        VECTOR_DB_FOLDER=os.path.join(app.instance_path, 'vector_dbs'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
    )
    
    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
    
    # Ensure required folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VECTOR_DB_FOLDER'], exist_ok=True)
    
    # Initialize the database with the app
    db.init_app(app)
    
    # Create thread pool executor for background tasks
    app.executor = ThreadPoolExecutor(max_workers=2)
    
    # Register blueprints
    from easy_rag.routes import main, document, vector_db, retriever, query, dependency, llm, settings, help, docs
    app.register_blueprint(main.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(vector_db.bp)
    app.register_blueprint(retriever.bp)
    app.register_blueprint(query.bp)
    app.register_blueprint(dependency.bp)
    app.register_blueprint(llm.bp)
    app.register_blueprint(settings.bp)
    app.register_blueprint(help.bp)
    app.register_blueprint(docs.bp)
    
    # Register error handlers
    from easy_rag import errors
    errors.init_app(app)
    
    # Register CLI commands
    from easy_rag import commands
    commands.init_app(app)
    
    # Add URL rule for the index page
    app.add_url_rule('/', endpoint='index', view_func=main.index)
    
    # Register template filters
    @app.template_filter('datetime')
    def format_datetime(timestamp):
        """Format a timestamp to a readable datetime string."""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    @app.template_filter('filesizeformat')
    def filesizeformat_filter(size):
        """Format file size to human-readable format."""
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        size = float(size)
        unit_index = 0
        
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
            
        return f"{size:.2f} {units[unit_index]}"
    
    @app.template_filter('tojson')
    def tojson_filter(obj, indent=None):
        """Convert object to JSON string."""
        import json
        
        # Custom JSON encoder to handle SQLAlchemy objects
        class SQLAlchemyEncoder(json.JSONEncoder):
            def default(self, obj):
                # Handle SQLAlchemy objects with __json__ method
                if hasattr(obj, '__json__'):
                    return obj.__json__()
                # Handle SQLAlchemy objects with to_dict method
                elif hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                # Handle SQLAlchemy MetaData objects
                elif str(obj.__class__.__name__) == 'MetaData':
                    return str(obj)
                # Handle datetime objects
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return super().default(obj)
        
        try:
            return json.dumps(obj, indent=indent, cls=SQLAlchemyEncoder)
        except TypeError:
            # Fallback to string representation if JSON serialization fails
            return json.dumps(str(obj), indent=indent)
    
    @app.template_filter('nl2br')
    def nl2br_filter(text):
        """Convert newlines to HTML line breaks."""
        if not text:
            return ""
        return text.replace('\n', '<br>')
    
    return app