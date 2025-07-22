import click
from flask.cli import with_appcontext
from easy_rag import db
from easy_rag.utils.dependency_manager import DependencyManager
from easy_rag.utils.migrations import migration_manager
import os

def init_app(app):
    """Register CLI commands with the app"""
    app.cli.add_command(init_db_command)
    app.cli.add_command(check_dependencies_command)
    app.cli.add_command(install_dependencies_command)
    app.cli.add_command(generate_requirements_command)
    app.cli.add_command(db_migrate_command)
    app.cli.add_command(db_rollback_command)
    app.cli.add_command(db_create_migration_command)
    app.cli.add_command(db_status_command)

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Initialize the database."""
    db.create_all()
    click.echo('Initialized the database.')

@click.command('check-dependencies')
@with_appcontext
def check_dependencies_command():
    """Check if all dependencies are installed."""
    core_deps = DependencyManager.check_core_dependencies()
    
    click.echo('Core dependencies:')
    for dep, installed in core_deps.items():
        status = 'Installed' if installed else 'Missing'
        click.echo(f'  {dep}: {status}')

@click.command('install-dependencies')
@click.option('--feature', help='Install dependencies for a specific feature')
@with_appcontext
def install_dependencies_command(feature):
    """Install dependencies."""
    if feature:
        click.echo(f'Installing dependencies for feature: {feature}')
        results = DependencyManager.install_feature_dependencies(feature)
    else:
        click.echo('Installing core dependencies')
        results = DependencyManager.install_core_dependencies()
    
    for dep, (success, message) in results.items():
        click.echo(f'  {dep}: {message}')

@click.command('generate-requirements')
@click.option('--output', default='requirements.txt', help='Output file path')
@with_appcontext
def generate_requirements_command(output):
    """Generate requirements.txt file."""
    success = DependencyManager.generate_requirements_file(output)
    if success:
        click.echo(f'Requirements file generated at {output}')
    else:
        click.echo('Failed to generate requirements file')

@click.command('db-migrate')
@with_appcontext
def db_migrate_command():
    """Apply all pending database migrations."""
    try:
        applied = migration_manager.migrate()
        if applied:
            click.echo(f"Applied {len(applied)} migrations:")
            for migration_id in applied:
                click.echo(f"  - {migration_id}")
        else:
            click.echo("No pending migrations to apply.")
    except Exception as e:
        click.echo(f"Error applying migrations: {str(e)}", err=True)

@click.command('db-rollback')
@click.option('--steps', default=1, help='Number of migrations to roll back')
@with_appcontext
def db_rollback_command(steps):
    """Roll back the last n migrations."""
    try:
        rolled_back = migration_manager.rollback(steps)
        if rolled_back:
            click.echo(f"Rolled back {len(rolled_back)} migrations:")
            for migration_id in rolled_back:
                click.echo(f"  - {migration_id}")
        else:
            click.echo("No migrations to roll back.")
    except Exception as e:
        click.echo(f"Error rolling back migrations: {str(e)}", err=True)

@click.command('db-create-migration')
@click.argument('name')
@with_appcontext
def db_create_migration_command(name):
    """Create a new migration file."""
    try:
        filepath = migration_manager.create_migration(name)
        click.echo(f"Created migration file: {filepath}")
        click.echo("Edit this file to add your migration SQL statements.")
    except Exception as e:
        click.echo(f"Error creating migration: {str(e)}", err=True)

@click.command('db-status')
@with_appcontext
def db_status_command():
    """Show status of database migrations."""
    try:
        # Show applied migrations
        applied = migration_manager.get_applied_migrations()
        click.echo(f"Applied migrations ({len(applied)}):")
        for migration in applied:
            click.echo(f"  - {migration['id']} ({migration['name']}) - {migration['applied_at']}")
        
        # Show pending migrations
        pending = migration_manager.get_pending_migrations()
        click.echo(f"\nPending migrations ({len(pending)}):")
        for migration_id in sorted(pending):
            click.echo(f"  - {migration_id}")
    except Exception as e:
        click.echo(f"Error getting migration status: {str(e)}", err=True)