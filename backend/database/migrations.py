"""Database migration scripts and utilities."""

import logging
from typing import List, Callable
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session

from .config import engine, Base, SessionLocal
from .models import Game, Event, TMICalculation

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, description: str, up_func: Callable, down_func: Callable = None):
        self.version = version
        self.description = description
        self.up_func = up_func
        self.down_func = down_func


def create_migration_table():
    """Create the migrations tracking table."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()


def get_applied_migrations() -> List[str]:
    """Get list of applied migration versions."""
    create_migration_table()
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version FROM schema_migrations ORDER BY version"))
        return [row[0] for row in result.fetchall()]


def mark_migration_applied(version: str, description: str):
    """Mark a migration as applied."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO schema_migrations (version, description) 
            VALUES (:version, :description)
        """), {"version": version, "description": description})
        conn.commit()


def mark_migration_reverted(version: str):
    """Mark a migration as reverted."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM schema_migrations WHERE version = :version"), {"version": version})
        conn.commit()


# Migration functions
def migration_001_initial_schema():
    """Create initial database schema."""
    logger.info("Creating initial database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Initial schema created successfully")


def migration_002_add_indexes():
    """Add performance indexes."""
    logger.info("Adding performance indexes...")
    
    with engine.connect() as conn:
        # Add indexes for common query patterns
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_game_timestamp ON events(game_id, timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_team_period ON events(team_tricode, period)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tmi_game_team ON tmi_calculations(game_id, team_tricode)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tmi_calculated_at ON tmi_calculations(calculated_at)"))
        conn.commit()
    
    logger.info("Performance indexes added successfully")


def migration_003_add_event_description():
    """Add description column to events table if it doesn't exist."""
    logger.info("Checking for description column in events table...")
    
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('events')]
    
    if 'description' not in columns:
        logger.info("Adding description column to events table...")
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE events ADD COLUMN description TEXT"))
            conn.commit()
        logger.info("Description column added successfully")
    else:
        logger.info("Description column already exists")


# Define all migrations
MIGRATIONS = [
    Migration("001", "Initial database schema", migration_001_initial_schema),
    Migration("002", "Add performance indexes", migration_002_add_indexes),
    Migration("003", "Add event description column", migration_003_add_event_description),
]


def run_migrations():
    """Run all pending migrations."""
    logger.info("Starting database migrations...")
    
    applied_migrations = get_applied_migrations()
    pending_migrations = [m for m in MIGRATIONS if m.version not in applied_migrations]
    
    if not pending_migrations:
        logger.info("No pending migrations")
        return
    
    for migration in pending_migrations:
        logger.info(f"Running migration {migration.version}: {migration.description}")
        try:
            migration.up_func()
            mark_migration_applied(migration.version, migration.description)
            logger.info(f"Migration {migration.version} completed successfully")
        except Exception as e:
            logger.error(f"Migration {migration.version} failed: {str(e)}")
            raise
    
    logger.info("All migrations completed successfully")


def rollback_migration(version: str):
    """Rollback a specific migration."""
    migration = next((m for m in MIGRATIONS if m.version == version), None)
    
    if not migration:
        raise ValueError(f"Migration {version} not found")
    
    if not migration.down_func:
        raise ValueError(f"Migration {version} does not support rollback")
    
    logger.info(f"Rolling back migration {version}: {migration.description}")
    
    try:
        migration.down_func()
        mark_migration_reverted(version)
        logger.info(f"Migration {version} rolled back successfully")
    except Exception as e:
        logger.error(f"Rollback of migration {version} failed: {str(e)}")
        raise


def reset_database():
    """Reset the entire database (drop and recreate all tables)."""
    logger.warning("Resetting database - all data will be lost!")
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    
    # Drop migration table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS schema_migrations"))
        conn.commit()
    
    # Run all migrations
    run_migrations()
    
    logger.info("Database reset completed")


def check_database_health() -> bool:
    """Check if database is accessible and has correct schema."""
    try:
        with SessionLocal() as db:
            # Try to query each table
            db.query(Game).first()
            db.query(Event).first()
            db.query(TMICalculation).first()
        
        logger.info("Database health check passed")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run migrations
    run_migrations()