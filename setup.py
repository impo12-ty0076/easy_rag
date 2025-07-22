"""
Setup script for Easy RAG System.
This script sets up the virtual environment and initializes the database.
"""
import os
import subprocess
import sys
import venv
import argparse
from pathlib import Path

def create_virtual_environment(venv_path):
    """Create a virtual environment at the specified path."""
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    print("Virtual environment created successfully.")

def install_dependencies(venv_path):
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    # Determine the path to the Python executable in the virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    # Install dependencies
    subprocess.check_call([python_path, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("Dependencies installed successfully.")

def initialize_database(venv_path):
    """Initialize the database."""
    print("Initializing database...")
    
    # Determine the path to the Python executable in the virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    # Run Flask command to initialize the database
    subprocess.check_call([python_path, 'init_db.py'])
    print("Database initialized successfully.")

def setup_project_structure():
    """Set up the project directory structure."""
    print("Setting up project structure...")
    
    # Define the directories to create
    directories = [
        'easy_rag',
        'easy_rag/routes',
        'easy_rag/static',
        'easy_rag/static/css',
        'easy_rag/static/js',
        'easy_rag/static/img',
        'easy_rag/templates',
        'easy_rag/templates/document',
        'easy_rag/templates/vector_db',
        'easy_rag/templates/retriever',
        'easy_rag/templates/query',
        'easy_rag/utils',
        'easy_rag/utils/db',
        'instance',
        'instance/uploads',
        'instance/vector_dbs',
    ]
    
    # Create the directories
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    print("Project structure setup completed.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Set up Easy RAG System.')
    parser.add_argument('--venv', default='venv', help='Path to virtual environment')
    parser.add_argument('--skip-venv', action='store_true', help='Skip virtual environment creation')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-db', action='store_true', help='Skip database initialization')
    args = parser.parse_args()
    
    venv_path = args.venv
    
    # Set up project structure
    setup_project_structure()
    
    # Create virtual environment
    if not args.skip_venv:
        if not os.path.exists(venv_path):
            create_virtual_environment(venv_path)
        else:
            print(f"Virtual environment already exists at {venv_path}")
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies(venv_path)
    
    # Initialize database
    if not args.skip_db:
        initialize_database(venv_path)
    
    print("\nSetup completed successfully!")
    print(f"To activate the virtual environment, run:")
    if os.name == 'nt':  # Windows
        print(f"    {venv_path}\\Scripts\\activate.bat")
    else:  # Unix/Linux/Mac
        print(f"    source {venv_path}/bin/activate")
    print("To run the application, activate the virtual environment and run:")
    print("    python app.py")

if __name__ == '__main__':
    main()