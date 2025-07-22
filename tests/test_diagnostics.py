import pytest
import os
import json
from flask import url_for
from easy_rag.utils.diagnostics import SystemDiagnostics

def test_system_health(app_context):
    """Test getting system health information."""
    # Get system health
    health_info = SystemDiagnostics.get_system_health()
    
    # Check that the health info contains the expected sections
    assert 'system' in health_info
    assert 'resources' in health_info
    assert 'python' in health_info
    assert 'database' in health_info
    assert 'dependencies' in health_info
    assert 'storage' in health_info
    assert 'errors' in health_info
    assert 'status' in health_info
    assert 'issues' in health_info
    
    # Check that the system info contains expected keys
    assert 'platform' in health_info['system']
    assert 'system' in health_info['system']
    assert 'release' in health_info['system']
    
    # Check that the resources info contains expected keys
    assert 'cpu_percent' in health_info['resources']
    assert 'memory_percent' in health_info['resources']
    assert 'disk_percent' in health_info['resources']

def test_run_diagnostics(app_context):
    """Test running system diagnostics."""
    # Run diagnostics
    diagnostics = SystemDiagnostics.run_diagnostics()
    
    # Check that the diagnostics contains the expected sections
    assert 'timestamp' in diagnostics
    assert 'health' in diagnostics
    assert 'tests' in diagnostics
    assert 'execution_time' in diagnostics
    
    # Check that the tests section contains the expected tests
    assert 'database_connectivity' in diagnostics['tests']
    assert 'file_system_access' in diagnostics['tests']
    assert 'dependency_installation' in diagnostics['tests']
    
    # Check that each test has the expected structure
    for test_name, test_result in diagnostics['tests'].items():
        assert 'name' in test_result
        assert 'success' in test_result
        assert 'message' in test_result
        assert 'details' in test_result

def test_diagnostics_route(client):
    """Test the diagnostics route."""
    # Get the diagnostics page
    response = client.get('/settings/diagnostics')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the page contains expected sections
    assert b'System Information' in response.data
    assert b'Resource Usage' in response.data
    assert b'Python Information' in response.data
    assert b'Database Health' in response.data
    assert b'Storage Health' in response.data
    assert b'Dependency Status' in response.data

def test_run_diagnostics_route(client):
    """Test the run diagnostics route."""
    # Run diagnostics
    response = client.get('/settings/diagnostics/run')
    
    # Check that the response is a redirect
    assert response.status_code == 302
    
    # Follow the redirect to the results page
    response = client.get('/settings/diagnostics/results')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Check that the page contains expected sections
    assert b'Diagnostic Results' in response.data
    assert b'System Health' in response.data
    assert b'Test Results' in response.data

def test_api_system_health(client):
    """Test the API endpoint for system health."""
    # Get system health from API
    response = client.get('/settings/api/system-health')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the JSON response
    data = json.loads(response.data)
    
    # Check that the response contains expected sections
    assert 'system' in data
    assert 'resources' in data
    assert 'python' in data
    assert 'database' in data
    assert 'dependencies' in data
    assert 'storage' in data
    assert 'errors' in data
    assert 'status' in data
    assert 'issues' in data

def test_api_dependency_status(client):
    """Test the API endpoint for dependency status."""
    # Get dependency status from API
    response = client.get('/settings/api/dependency-status')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the JSON response
    data = json.loads(response.data)
    
    # Check that the response contains dependency information
    assert isinstance(data, dict)
    assert len(data) > 0

def test_api_error_logs(client, app):
    """Test the API endpoint for error logs."""
    # Create a test log file
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'easy_rag.log')
    with open(log_file, 'w') as f:
        f.write('2023-01-01 12:00:00 ERROR Test error message\n')
        f.write('2023-01-01 12:01:00 INFO Test info message\n')
        f.write('2023-01-01 12:02:00 ERROR Another test error message\n')
    
    # Get error logs from API
    response = client.get('/settings/api/error-logs')
    
    # Check that the response is successful
    assert response.status_code == 200
    
    # Parse the JSON response
    data = json.loads(response.data)
    
    # Check that the response contains error logs
    assert isinstance(data, list)
    assert len(data) == 2  # Should have 2 error messages
    assert 'ERROR Test error message' in data[0]
    assert 'ERROR Another test error message' in data[1]
    
    # Clean up
    os.remove(log_file)