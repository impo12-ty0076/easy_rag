"""
Dependency management utilities for Easy RAG System.
This module provides tools for checking and installing dependencies.
"""
import subprocess
import sys
import importlib
import pkg_resources
import logging
import time
import platform
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import threading
import json
from flask import current_app

# Set up logging
logger = logging.getLogger(__name__)

class DependencyManager:
    """Class for managing dependencies in the Easy RAG System."""
    
    # Core dependencies required for the application to function
    CORE_DEPENDENCIES = {
        "flask": "2.0.0",
        "sqlalchemy": "1.4.0",
        "langchain": "0.0.267",
        "click": "8.0.0",
        "python-dotenv": "0.19.0"
    }
    
    # Feature-specific dependencies
    FEATURE_DEPENDENCIES = {
        "document_loaders": {
            "pdf": {
                "pypdf": "3.15.1",
                "pdfminer.six": "20221105"
            },
            "docx": {
                "python-docx": "0.8.11"
            },
            "csv": {
                "pandas": "2.0.0"
            },
            "markdown": {
                "markdown": "3.4.3"
            },
            "html": {
                "beautifulsoup4": "4.12.2",
                "html5lib": "1.1"
            }
        },
        "embedding_models": {
            "sentence_transformers": {
                "sentence-transformers": "2.2.2",
                "torch": "2.0.0"
            },
            "openai": {
                "openai": "0.28.0"
            },
            "huggingface": {
                "transformers": "4.30.0",
                "torch": "2.0.0"
            }
        },
        "vector_stores": {
            "chroma": {
                "chromadb": "0.4.6"
            },
            "faiss": {
                "faiss-cpu": "1.7.4"
            },
            "pinecone": {
                "pinecone-client": "2.2.2"
            }
        },
        "llms": {
            "openai": {
                "openai": "0.28.0"
            },
            "huggingface": {
                "transformers": "4.30.0",
                "torch": "2.0.0",
                "accelerate": "0.21.0"
            },
            "llama_cpp": {
                "llama-cpp-python": "0.1.77"
            }
        }
    }
    
    # Installation status constants
    STATUS_NOT_STARTED = "not_started"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETE = "complete"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    
    # Progress tracking for installations
    _installation_progress = {}
    _installation_lock = threading.Lock()
    _installation_history = []
    _max_history_entries = 10
    
    # System information
    _system_info = None
    
    @classmethod
    def check_core_dependencies(cls) -> Dict[str, bool]:
        """
        Check if core dependencies are installed.
        
        Returns:
            Dict[str, bool]: Dictionary mapping dependency names to installation status
        """
        results = {}
        for package, version in cls.CORE_DEPENDENCIES.items():
            results[package] = cls._is_package_installed(package, version)
        return results
    
    @classmethod
    def check_feature_dependencies(cls, feature: str) -> Dict[str, bool]:
        """
        Check if dependencies for a specific feature are installed.
        
        Args:
            feature (str): Feature name in format 'category/feature' (e.g., 'document_loaders/pdf')
            
        Returns:
            Dict[str, bool]: Dictionary mapping dependency names to installation status
        """
        category, feature_name = cls._parse_feature_string(feature)
        if not category or not feature_name:
            logger.error(f"Invalid feature format: {feature}")
            return {}
        
        try:
            dependencies = cls.FEATURE_DEPENDENCIES[category][feature_name]
            results = {}
            for package, version in dependencies.items():
                results[package] = cls._is_package_installed(package, version)
            return results
        except KeyError:
            logger.error(f"Unknown feature: {feature}")
            return {}
    
    @classmethod
    def install_core_dependencies(cls) -> Dict[str, Tuple[bool, str]]:
        """
        Install core dependencies.
        
        Returns:
            Dict[str, Tuple[bool, str]]: Dictionary mapping dependency names to (success, message) tuples
        """
        results = {}
        total_deps = len(cls.CORE_DEPENDENCIES)
        
        # Initialize progress tracking
        with cls._installation_lock:
            cls._installation_progress = {
                "total": total_deps,
                "completed": 0,
                "current_package": "",
                "status": "in_progress",
                "errors": []
            }
        
        for i, (package, version) in enumerate(cls.CORE_DEPENDENCIES.items()):
            # Update progress
            with cls._installation_lock:
                cls._installation_progress["current_package"] = package
            
            # Install package
            success, message = cls._install_package(package, version)
            results[package] = (success, message)
            
            # Update progress
            with cls._installation_lock:
                cls._installation_progress["completed"] = i + 1
                if not success:
                    cls._installation_progress["errors"].append(f"{package}: {message}")
        
        # Mark installation as complete
        with cls._installation_lock:
            cls._installation_progress["status"] = "complete"
        
        return results
    
    @classmethod
    def install_feature_dependencies(cls, feature: str) -> Dict[str, Tuple[bool, str]]:
        """
        Install dependencies for a specific feature.
        
        Args:
            feature (str): Feature name in format 'category/feature' (e.g., 'document_loaders/pdf')
            
        Returns:
            Dict[str, Tuple[bool, str]]: Dictionary mapping dependency names to (success, message) tuples
        """
        category, feature_name = cls._parse_feature_string(feature)
        if not category or not feature_name:
            logger.error(f"Invalid feature format: {feature}")
            return {"error": (False, f"Invalid feature format: {feature}")}
        
        try:
            dependencies = cls.FEATURE_DEPENDENCIES[category][feature_name]
        except KeyError:
            logger.error(f"Unknown feature: {feature}")
            return {"error": (False, f"Unknown feature: {feature}")}
        
        results = {}
        total_deps = len(dependencies)
        
        # Initialize progress tracking
        with cls._installation_lock:
            cls._installation_progress = {
                "total": total_deps,
                "completed": 0,
                "current_package": "",
                "status": "in_progress",
                "feature": feature,
                "errors": []
            }
        
        for i, (package, version) in enumerate(dependencies.items()):
            # Update progress
            with cls._installation_lock:
                cls._installation_progress["current_package"] = package
            
            # Install package
            success, message = cls._install_package(package, version)
            results[package] = (success, message)
            
            # Update progress
            with cls._installation_lock:
                cls._installation_progress["completed"] = i + 1
                if not success:
                    cls._installation_progress["errors"].append(f"{package}: {message}")
        
        # Mark installation as complete
        with cls._installation_lock:
            cls._installation_progress["status"] = "complete"
        
        return results
    
    @classmethod
    def get_installation_progress(cls) -> Dict[str, Any]:
        """
        Get the current progress of package installation.
        
        Returns:
            Dict[str, Any]: Dictionary with progress information
        """
        with cls._installation_lock:
            # Create a copy to avoid race conditions
            return dict(cls._installation_progress)
    
    @classmethod
    def generate_requirements_file(cls, output_path: str) -> bool:
        """
        Generate a requirements.txt file with all dependencies.
        
        Args:
            output_path (str): Path to output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Collect all dependencies
            all_deps = {}
            all_deps.update(cls.CORE_DEPENDENCIES)
            
            # Add feature dependencies
            for category, features in cls.FEATURE_DEPENDENCIES.items():
                for feature, deps in features.items():
                    all_deps.update(deps)
            
            # Write to file
            with open(output_path, 'w') as f:
                for package, version in sorted(all_deps.items()):
                    f.write(f"{package}>={version}\n")
            
            return True
        except Exception as e:
            logger.error(f"Error generating requirements file: {str(e)}")
            return False
    
    @classmethod
    def _is_package_installed(cls, package: str, min_version: str) -> bool:
        """
        Check if a package is installed with at least the specified version.
        
        Args:
            package (str): Package name
            min_version (str): Minimum version required
            
        Returns:
            bool: True if installed with required version, False otherwise
        """
        try:
            # Try to import the package
            importlib.import_module(package.replace('-', '_'))
            
            # Check version
            installed_version = pkg_resources.get_distribution(package).version
            
            # Compare versions
            return pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version)
        except (ImportError, pkg_resources.DistributionNotFound):
            return False
    
    @classmethod
    def _install_package(cls, package: str, version: str) -> Tuple[bool, str]:
        """
        Install a package using pip.
        
        Args:
            package (str): Package name
            version (str): Version to install
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Check if already installed with correct version
            if cls._is_package_installed(package, version):
                return True, f"Already installed (>= {version})"
            
            # Install package
            cmd = [sys.executable, "-m", "pip", "install", f"{package}>={version}"]
            logger.info(f"Installing {package}>={version}")
            
            # Run pip install
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Track installation progress with periodic updates
            start_time = time.time()
            last_update_time = start_time
            
            # Wait for process to complete with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
                
                if process.returncode == 0:
                    # Record successful installation in history
                    cls._add_to_installation_history(package, version, True, f"Successfully installed {package}>={version}")
                    return True, f"Successfully installed {package}>={version}"
                else:
                    error_msg = stderr.strip() or "Unknown error"
                    # Record failed installation in history
                    cls._add_to_installation_history(package, version, False, error_msg)
                    logger.error(f"Failed to install {package}: {error_msg}")
                    return False, error_msg
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                stdout, stderr = process.communicate()
                error_msg = "Installation timed out after 5 minutes"
                cls._add_to_installation_history(package, version, False, error_msg)
                logger.error(f"Timeout installing {package}: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = str(e)
            cls._add_to_installation_history(package, version, False, error_msg)
            logger.error(f"Error installing {package}: {error_msg}")
            return False, error_msg
    
    @classmethod
    def _parse_feature_string(cls, feature: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a feature string in the format 'category/feature'.
        
        Args:
            feature (str): Feature string (e.g., 'document_loaders/pdf')
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (category, feature_name) or (None, None) if invalid
        """
        parts = feature.split('/')
        if len(parts) != 2:
            return None, None
        
        category, feature_name = parts
        
        # Validate category and feature exist
        if category not in cls.FEATURE_DEPENDENCIES:
            return None, None
        
        if feature_name not in cls.FEATURE_DEPENDENCIES[category]:
            return None, None
        
        return category, feature_name
        
    @classmethod
    def _add_to_installation_history(cls, package: str, version: str, success: bool, message: str) -> None:
        """
        Add an installation attempt to the history.
        
        Args:
            package (str): Package name
            version (str): Version
            success (bool): Whether installation was successful
            message (str): Installation message or error
        """
        with cls._installation_lock:
            # Add to history
            cls._installation_history.append({
                "package": package,
                "version": version,
                "success": success,
                "message": message,
                "timestamp": time.time()
            })
            
            # Trim history if needed
            if len(cls._installation_history) > cls._max_history_entries:
                cls._installation_history = cls._installation_history[-cls._max_history_entries:]
    
    @classmethod
    def get_installation_history(cls) -> List[Dict[str, Any]]:
        """
        Get the installation history.
        
        Returns:
            List[Dict[str, Any]]: List of installation attempts
        """
        with cls._installation_lock:
            # Create a copy to avoid race conditions
            return list(cls._installation_history)
    
    @classmethod
    def cancel_installation(cls) -> bool:
        """
        Cancel the current installation process.
        
        Returns:
            bool: True if cancelled, False if no installation in progress
        """
        with cls._installation_lock:
            if not cls._installation_progress or cls._installation_progress.get("status") != cls.STATUS_IN_PROGRESS:
                return False
            
            cls._installation_progress["status"] = cls.STATUS_CANCELLED
            return True
    
    @classmethod
    def get_system_info(cls) -> Dict[str, str]:
        """
        Get system information.
        
        Returns:
            Dict[str, str]: Dictionary with system information
        """
        if cls._system_info is None:
            cls._system_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "system": platform.system(),
                "python_path": sys.executable,
                "pip_version": cls._get_pip_version(),
                "user_site_packages": cls._get_site_packages_path()
            }
        
        return cls._system_info
    
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
    def _get_site_packages_path(cls) -> str:
        """
        Get the site-packages directory path.
        
        Returns:
            str: Path to site-packages
        """
        try:
            import site
            return site.getsitepackages()[0]
        except Exception:
            return "Unknown"
    
    @classmethod
    def get_dependency_details(cls, package_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a dependency.
        
        Args:
            package_name (str): Package name
            
        Returns:
            Dict[str, Any]: Dictionary with package details
        """
        try:
            # Check if package is installed
            try:
                dist = pkg_resources.get_distribution(package_name)
                installed = True
                version = dist.version
                location = dist.location
                requires = [str(r) for r in dist.requires()]
            except pkg_resources.DistributionNotFound:
                installed = False
                version = None
                location = None
                requires = []
            
            # Check if package is in core dependencies
            in_core = package_name in cls.CORE_DEPENDENCIES
            core_version = cls.CORE_DEPENDENCIES.get(package_name)
            
            # Check if package is in feature dependencies
            features = []
            for category, category_features in cls.FEATURE_DEPENDENCIES.items():
                for feature_name, deps in category_features.items():
                    if package_name in deps:
                        features.append(f"{category}/{feature_name}")
            
            return {
                "name": package_name,
                "installed": installed,
                "version": version,
                "location": location,
                "requires": requires,
                "in_core": in_core,
                "core_version": core_version,
                "features": features
            }
        except Exception as e:
            logger.error(f"Error getting dependency details for {package_name}: {str(e)}")
            return {
                "name": package_name,
                "error": str(e)
            }
    
    @classmethod
    def get_all_dependencies(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get details for all dependencies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping package names to details
        """
        all_deps = {}
        
        # Add core dependencies
        for package, version in cls.CORE_DEPENDENCIES.items():
            all_deps[package] = cls.get_dependency_details(package)
        
        # Add feature dependencies
        for category, features in cls.FEATURE_DEPENDENCIES.items():
            for feature_name, deps in features.items():
                for package, version in deps.items():
                    if package not in all_deps:
                        all_deps[package] = cls.get_dependency_details(package)
        
        return all_deps
    
    @classmethod
    def diagnose_installation_issues(cls) -> Dict[str, Any]:
        """
        Diagnose common installation issues.
        
        Returns:
            Dict[str, Any]: Dictionary with diagnostic information
        """
        diagnostics = {
            "system_info": cls.get_system_info(),
            "issues": [],
            "recommendations": []
        }
        
        # Check for common issues
        try:
            # Check if pip is working
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if pip_result.returncode != 0:
                diagnostics["issues"].append("Pip is not working properly")
                diagnostics["recommendations"].append("Reinstall pip or Python")
            
            # Check if site-packages is writable
            site_packages = cls._get_site_packages_path()
            if site_packages != "Unknown":
                if not os.access(site_packages, os.W_OK):
                    diagnostics["issues"].append("Site-packages directory is not writable")
                    diagnostics["recommendations"].append("Run with elevated permissions or use a virtual environment")
            
            # Check for network connectivity
            try:
                # Try to connect to PyPI
                import socket
                socket.create_connection(("pypi.org", 443), timeout=5)
            except Exception:
                diagnostics["issues"].append("Network connectivity issues detected")
                diagnostics["recommendations"].append("Check your internet connection or proxy settings")
            
            # Check for conflicting packages
            # This is a simplified check - in a real system you might want more sophisticated checks
            core_deps = cls.check_core_dependencies()
            if not all(core_deps.values()):
                missing_deps = [pkg for pkg, installed in core_deps.items() if not installed]
                diagnostics["issues"].append(f"Missing core dependencies: {', '.join(missing_deps)}")
                diagnostics["recommendations"].append("Run 'flask install-dependencies' to install missing dependencies")
            
        except Exception as e:
            diagnostics["issues"].append(f"Error during diagnostics: {str(e)}")
        
        return diagnostics