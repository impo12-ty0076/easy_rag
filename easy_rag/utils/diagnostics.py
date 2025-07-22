"""
System diagnostics utilities for Easy RAG System.
This module provides tools for checking system health and diagnosing issues.
"""
import os
import sys
import platform
import subprocess
import logging
import time

# Try to import psutil, but provide fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, List, Any, Optional
import sqlite3
from flask import current_app
from easy_rag.utils.dependency_manager import DependencyManager

# Set up logging
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """Class for system diagnostics in the Easy RAG System."""
    
    @classmethod
    def get_system_health(cls) -> Dict[str, Any]:
        """
        Get comprehensive system health information.
        
        Returns:
            Dict[str, Any]: Dictionary with system health information
        """
        health_info = {
            'system': cls._get_system_info(),
            'resources': cls._get_resource_usage(),
            'python': cls._get_python_info(),
            'database': cls._check_database_health(),
            'dependencies': cls._check_dependency_health(),
            'storage': cls._check_storage_health(),
            'errors': cls._get_recent_errors()
        }
        
        # Calculate overall health status
        health_status = 'healthy'
        issues = []
        
        # Check database health
        if not health_info['database']['connected']:
            health_status = 'critical'
            issues.append('Database connection failed')
        
        # Check storage health
        if health_info['storage']['disk_space_percent'] > 90:
            health_status = 'warning' if health_status == 'healthy' else health_status
            issues.append('Disk space is low')
        
        # Check memory usage
        if health_info['resources']['memory_percent'] > 90:
            health_status = 'warning' if health_status == 'healthy' else health_status
            issues.append('Memory usage is high')
        
        # Check dependency health
        missing_deps = [dep for dep, installed in health_info['dependencies']['core_dependencies'].items() if not installed]
        if missing_deps:
            health_status = 'warning' if health_status == 'healthy' else health_status
            issues.append(f'Missing core dependencies: {", ".join(missing_deps)}')
        
        # Add overall status
        health_info['status'] = health_status
        health_info['issues'] = issues
        
        return health_info
    
    @classmethod
    def _get_system_info(cls) -> Dict[str, str]:
        """
        Get basic system information.
        
        Returns:
            Dict[str, str]: Dictionary with system information
        """
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node()
        }
    
    @classmethod
    def _get_resource_usage(cls) -> Dict[str, Any]:
        """
        Get system resource usage information.
        
        Returns:
            Dict[str, Any]: Dictionary with resource usage information
        """
        result = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_used': 0,
            'memory_total': 0,
            'disk_percent': 0,
            'disk_used': 0,
            'disk_total': 0
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # Get CPU usage
                result['cpu_percent'] = psutil.cpu_percent(interval=0.5)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                result['memory_percent'] = memory.percent
                result['memory_used'] = memory.used
                result['memory_total'] = memory.total
                
                # Get disk usage for the current directory
                disk = psutil.disk_usage(os.getcwd())
                result['disk_percent'] = disk.percent
                result['disk_used'] = disk.used
                result['disk_total'] = disk.total
            except Exception as e:
                logger.error(f"Error getting resource usage with psutil: {str(e)}")
        else:
            # Fallback to basic information if psutil is not available
            logger.warning("psutil not available, using fallback for resource usage")
            
            # Try to get disk information using os.statvfs on Unix-like systems
            try:
                if hasattr(os, 'statvfs'):  # Unix/Linux/MacOS
                    statvfs = os.statvfs(os.getcwd())
                    disk_total = statvfs.f_frsize * statvfs.f_blocks
                    disk_free = statvfs.f_frsize * statvfs.f_bfree
                    disk_used = disk_total - disk_free
                    
                    result['disk_total'] = disk_total
                    result['disk_used'] = disk_used
                    result['disk_percent'] = (disk_used / disk_total) * 100 if disk_total > 0 else 0
            except Exception:
                pass
        
        return result
    
    @classmethod
    def _get_python_info(cls) -> Dict[str, str]:
        """
        Get Python-specific information.
        
        Returns:
            Dict[str, str]: Dictionary with Python information
        """
        return {
            'version': sys.version,
            'executable': sys.executable,
            'path': str(sys.path),
            'pip_version': cls._get_pip_version()
        }
    
    @classmethod
    def _get_pip_version(cls) -> str:
        """
        Get the pip version.
        
        Returns:
            str: Pip version
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
        except Exception:
            return "Error getting pip version"
    
    @classmethod
    def _check_database_health(cls) -> Dict[str, Any]:
        """
        Check database health.
        
        Returns:
            Dict[str, Any]: Dictionary with database health information
        """
        result = {
            'connected': False,
            'tables': [],
            'size': 0,
            'location': '',
            'errors': []
        }
        
        try:
            # Get database path from app config
            if current_app:
                db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
                result['location'] = db_path
                
                # Check if database file exists
                if os.path.exists(db_path):
                    result['size'] = os.path.getsize(db_path)
                    
                    # Connect to database and get table list
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    result['tables'] = [table[0] for table in tables]
                    conn.close()
                    
                    result['connected'] = True
                else:
                    result['errors'].append('Database file does not exist')
            else:
                result['errors'].append('Flask application context not available')
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    @classmethod
    def _check_dependency_health(cls) -> Dict[str, Any]:
        """
        Check dependency health.
        
        Returns:
            Dict[str, Any]: Dictionary with dependency health information
        """
        return {
            'core_dependencies': DependencyManager.check_core_dependencies(),
            'installation_history': DependencyManager.get_installation_history()
        }
    
    @classmethod
    def _check_storage_health(cls) -> Dict[str, Any]:
        """
        Check storage health.
        
        Returns:
            Dict[str, Any]: Dictionary with storage health information
        """
        result = {
            'disk_space_percent': 0,
            'disk_space_free': 0,
            'disk_space_total': 0,
            'upload_folder_size': 0,
            'vector_db_folder_size': 0,
            'log_folder_size': 0
        }
        
        try:
            # Get disk usage
            if PSUTIL_AVAILABLE:
                disk = psutil.disk_usage(os.getcwd())
                result['disk_space_percent'] = disk.percent
                result['disk_space_free'] = disk.free
                result['disk_space_total'] = disk.total
            else:
                # Fallback to basic information if psutil is not available
                logger.warning("psutil not available, using fallback for storage health")
                
                # Try to get disk information using os.statvfs on Unix-like systems
                try:
                    if hasattr(os, 'statvfs'):  # Unix/Linux/MacOS
                        statvfs = os.statvfs(os.getcwd())
                        disk_total = statvfs.f_frsize * statvfs.f_blocks
                        disk_free = statvfs.f_frsize * statvfs.f_bfree
                        disk_used = disk_total - disk_free
                        
                        result['disk_space_total'] = disk_total
                        result['disk_space_free'] = disk_free
                        result['disk_space_percent'] = (disk_used / disk_total) * 100 if disk_total > 0 else 0
                except Exception:
                    pass
            
            # Get folder sizes if app context is available
            if current_app:
                # Upload folder
                upload_folder = current_app.config.get('UPLOAD_FOLDER')
                if upload_folder and os.path.exists(upload_folder):
                    result['upload_folder_size'] = cls._get_folder_size(upload_folder)
                
                # Vector DB folder
                vector_db_folder = current_app.config.get('VECTOR_DB_FOLDER')
                if vector_db_folder and os.path.exists(vector_db_folder):
                    result['vector_db_folder_size'] = cls._get_folder_size(vector_db_folder)
            
            # Log folder
            log_folder = os.path.join(os.getcwd(), 'logs')
            if os.path.exists(log_folder):
                result['log_folder_size'] = cls._get_folder_size(log_folder)
        except Exception as e:
            logger.error(f"Error checking storage health: {str(e)}")
        
        return result
    
    @classmethod
    def _get_folder_size(cls, folder_path: str) -> int:
        """
        Calculate the total size of a folder in bytes.
        
        Args:
            folder_path (str): Path to the folder
            
        Returns:
            int: Folder size in bytes
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    
    @classmethod
    def _get_recent_errors(cls, max_entries: int = 10) -> List[str]:
        """
        Get recent error logs.
        
        Args:
            max_entries (int): Maximum number of log entries to return
            
        Returns:
            List[str]: List of error log entries
        """
        logs = []
        log_file = os.path.join('logs', 'easy_rag.log')
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Process lines to extract error logs
                    for line in lines[-1000:]:  # Look at last 1000 lines
                        if 'ERROR' in line:
                            logs.append(line.strip())
                            
                            # Limit the number of entries
                            if len(logs) >= max_entries:
                                break
            except Exception as e:
                logs.append(f"Error reading log file: {str(e)}")
        else:
            logs.append("Log file not found")
        
        return logs[-max_entries:]  # Return only the most recent entries
    
    @classmethod
    def run_diagnostics(cls) -> Dict[str, Any]:
        """
        Run comprehensive system diagnostics.
        
        Returns:
            Dict[str, Any]: Dictionary with diagnostic results
        """
        start_time = time.time()
        
        diagnostics = {
            'timestamp': time.time(),
            'health': cls.get_system_health(),
            'tests': {}
        }
        
        # Run database connectivity test
        diagnostics['tests']['database_connectivity'] = cls._test_database_connectivity()
        
        # Run file system access test
        diagnostics['tests']['file_system_access'] = cls._test_file_system_access()
        
        # Run dependency installation test
        diagnostics['tests']['dependency_installation'] = cls._test_dependency_installation()
        
        # Calculate execution time
        diagnostics['execution_time'] = time.time() - start_time
        
        return diagnostics
    
    @classmethod
    def _test_database_connectivity(cls) -> Dict[str, Any]:
        """
        Test database connectivity.
        
        Returns:
            Dict[str, Any]: Test results
        """
        result = {
            'name': 'Database Connectivity Test',
            'success': False,
            'message': '',
            'details': {}
        }
        
        try:
            # Get database path from app config
            if current_app:
                db_path = current_app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
                
                # Check if database file exists
                if os.path.exists(db_path):
                    # Try to connect and run a simple query
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT sqlite_version();")
                    version = cursor.fetchone()
                    conn.close()
                    
                    result['success'] = True
                    result['message'] = 'Successfully connected to database'
                    result['details'] = {
                        'path': db_path,
                        'sqlite_version': version[0] if version else 'Unknown'
                    }
                else:
                    result['message'] = 'Database file does not exist'
                    result['details'] = {'path': db_path}
            else:
                result['message'] = 'Flask application context not available'
        except Exception as e:
            result['message'] = f'Error connecting to database: {str(e)}'
        
        return result
    
    @classmethod
    def _test_file_system_access(cls) -> Dict[str, Any]:
        """
        Test file system access.
        
        Returns:
            Dict[str, Any]: Test results
        """
        result = {
            'name': 'File System Access Test',
            'success': False,
            'message': '',
            'details': {}
        }
        
        try:
            # Create a temporary file
            temp_file = os.path.join('logs', 'test_file.tmp')
            with open(temp_file, 'w') as f:
                f.write('Test file for file system access check')
            
            # Read the file
            with open(temp_file, 'r') as f:
                content = f.read()
            
            # Delete the file
            os.remove(temp_file)
            
            result['success'] = True
            result['message'] = 'Successfully created, read, and deleted test file'
            result['details'] = {
                'path': temp_file,
                'content_length': len(content)
            }
        except Exception as e:
            result['message'] = f'Error testing file system access: {str(e)}'
        
        return result
    
    @classmethod
    def _test_dependency_installation(cls) -> Dict[str, Any]:
        """
        Test dependency installation.
        
        Returns:
            Dict[str, Any]: Test results
        """
        result = {
            'name': 'Dependency Installation Test',
            'success': False,
            'message': '',
            'details': {}
        }
        
        try:
            # Check core dependencies
            core_deps = DependencyManager.check_core_dependencies()
            missing_deps = [dep for dep, installed in core_deps.items() if not installed]
            
            if missing_deps:
                result['message'] = f'Missing core dependencies: {", ".join(missing_deps)}'
                result['details'] = {'missing_dependencies': missing_deps}
            else:
                result['success'] = True
                result['message'] = 'All core dependencies are installed'
                result['details'] = {'dependencies': list(core_deps.keys())}
        except Exception as e:
            result['message'] = f'Error testing dependency installation: {str(e)}'
        
        return result