#!/usr/bin/env python3
"""Database management CLI script."""

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from database.migrations import run_migrations, reset_database, check_database_health
from database.config import create_tables, drop_tables


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='MomentumML Database Management')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Run database migrations')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database (WARNING: destroys all data)')
    reset_parser.add_argument('--confirm', action='store_true', required=True,
                             help='Confirm that you want to destroy all data')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check database health')
    
    # Create tables command
    create_parser = subparsers.add_parser('create', help='Create database tables')
    
    # Drop tables command
    drop_parser = subparsers.add_parser('drop', help='Drop database tables')
    drop_parser.add_argument('--confirm', action='store_true', required=True,
                            help='Confirm that you want to drop all tables')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'migrate':
            print("Running database migrations...")
            run_migrations()
            print("Migrations completed successfully!")
        
        elif args.command == 'reset':
            if not args.confirm:
                print("ERROR: --confirm flag is required for reset command")
                return 1
            
            print("WARNING: This will destroy all data in the database!")
            response = input("Type 'yes' to continue: ")
            if response.lower() != 'yes':
                print("Reset cancelled.")
                return 0
            
            print("Resetting database...")
            reset_database()
            print("Database reset completed!")
        
        elif args.command == 'health':
            print("Checking database health...")
            if check_database_health():
                print("✓ Database is healthy")
                return 0
            else:
                print("✗ Database health check failed")
                return 1
        
        elif args.command == 'create':
            print("Creating database tables...")
            create_tables()
            print("Tables created successfully!")
        
        elif args.command == 'drop':
            if not args.confirm:
                print("ERROR: --confirm flag is required for drop command")
                return 1
            
            print("WARNING: This will drop all tables!")
            response = input("Type 'yes' to continue: ")
            if response.lower() != 'yes':
                print("Drop cancelled.")
                return 0
            
            print("Dropping database tables...")
            drop_tables()
            print("Tables dropped successfully!")
    
    except Exception as e:
        logging.error(f"Command failed: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())