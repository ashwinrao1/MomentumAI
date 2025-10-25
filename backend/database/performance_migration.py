"""
Database migration for performance optimization indexes.

This migration adds indexes to improve query performance for frequently
accessed data patterns in the MomentumML application.
"""

import logging
from sqlalchemy import text
from .config import engine

logger = logging.getLogger("momentum_ml.migration.performance")


def create_performance_indexes():
    """Create performance optimization indexes."""
    
    indexes = [
        # Events table indexes
        {
            "name": "idx_events_game_timestamp",
            "table": "events",
            "columns": ["game_id", "timestamp"],
            "description": "Optimize queries for events by game and time"
        },
        {
            "name": "idx_events_game_team_timestamp", 
            "table": "events",
            "columns": ["game_id", "team_tricode", "timestamp"],
            "description": "Optimize queries for team-specific events"
        },
        {
            "name": "idx_events_type_timestamp",
            "table": "events", 
            "columns": ["event_type", "timestamp"],
            "description": "Optimize queries by event type"
        },
        
        # TMI calculations table indexes
        {
            "name": "idx_tmi_game_team_calculated",
            "table": "tmi_calculations",
            "columns": ["game_id", "team_tricode", "calculated_at"],
            "description": "Optimize TMI queries by game and team"
        },
        {
            "name": "idx_tmi_calculated_at",
            "table": "tmi_calculations",
            "columns": ["calculated_at"],
            "description": "Optimize time-based TMI queries"
        },
        {
            "name": "idx_tmi_game_calculated",
            "table": "tmi_calculations", 
            "columns": ["game_id", "calculated_at"],
            "description": "Optimize game-wide TMI queries"
        },
        
        # Games table indexes (additional)
        {
            "name": "idx_games_date_status",
            "table": "games",
            "columns": ["game_date", "status"],
            "description": "Optimize queries for games by date and status"
        }
    ]
    
    with engine.connect() as conn:
        for index in indexes:
            try:
                # Check if index already exists
                check_query = text(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND name='{index['name']}'
                """)
                
                result = conn.execute(check_query).fetchone()
                
                if result:
                    logger.info(f"Index {index['name']} already exists, skipping")
                    continue
                
                # Create the index
                columns_str = ", ".join(index['columns'])
                create_query = text(f"""
                    CREATE INDEX {index['name']} 
                    ON {index['table']} ({columns_str})
                """)
                
                conn.execute(create_query)
                conn.commit()
                
                logger.info(f"Created index {index['name']} on {index['table']}({columns_str})")
                
            except Exception as e:
                logger.error(f"Failed to create index {index['name']}: {e}")
                conn.rollback()
                continue
    
    logger.info("Performance indexes migration completed")


def analyze_table_statistics():
    """Analyze table statistics for query optimization."""
    
    tables = ["games", "events", "tmi_calculations"]
    
    with engine.connect() as conn:
        for table in tables:
            try:
                # Get row count
                count_query = text(f"SELECT COUNT(*) as count FROM {table}")
                count_result = conn.execute(count_query).fetchone()
                row_count = count_result[0] if count_result else 0
                
                # Get table info
                info_query = text(f"PRAGMA table_info({table})")
                columns = conn.execute(info_query).fetchall()
                
                logger.info(f"Table {table}: {row_count} rows, {len(columns)} columns")
                
                # For events table, get additional statistics
                if table == "events" and row_count > 0:
                    # Get event type distribution
                    type_query = text("""
                        SELECT event_type, COUNT(*) as count 
                        FROM events 
                        GROUP BY event_type 
                        ORDER BY count DESC 
                        LIMIT 10
                    """)
                    type_results = conn.execute(type_query).fetchall()
                    
                    logger.info(f"Top event types: {dict(type_results)}")
                    
                    # Get game distribution
                    game_query = text("""
                        SELECT COUNT(DISTINCT game_id) as unique_games 
                        FROM events
                    """)
                    game_result = conn.execute(game_query).fetchone()
                    unique_games = game_result[0] if game_result else 0
                    
                    logger.info(f"Events span {unique_games} unique games")
                
                # For TMI calculations, get team distribution
                elif table == "tmi_calculations" and row_count > 0:
                    team_query = text("""
                        SELECT team_tricode, COUNT(*) as count 
                        FROM tmi_calculations 
                        GROUP BY team_tricode 
                        ORDER BY count DESC 
                        LIMIT 10
                    """)
                    team_results = conn.execute(team_query).fetchall()
                    
                    logger.info(f"Top teams by TMI calculations: {dict(team_results)}")
                
            except Exception as e:
                logger.error(f"Failed to analyze table {table}: {e}")
                continue


def optimize_database_settings():
    """Apply database optimization settings for SQLite."""
    
    optimizations = [
        # Enable WAL mode for better concurrency
        "PRAGMA journal_mode=WAL",
        
        # Increase cache size (in pages, default page size is 4KB)
        "PRAGMA cache_size=10000",  # 40MB cache
        
        # Enable foreign key constraints
        "PRAGMA foreign_keys=ON",
        
        # Set synchronous mode for better performance
        "PRAGMA synchronous=NORMAL",
        
        # Optimize for faster writes
        "PRAGMA temp_store=MEMORY",
        
        # Set page size for better I/O
        "PRAGMA page_size=4096",
        
        # Enable automatic index creation
        "PRAGMA automatic_index=ON"
    ]
    
    with engine.connect() as conn:
        for pragma in optimizations:
            try:
                conn.execute(text(pragma))
                logger.info(f"Applied optimization: {pragma}")
            except Exception as e:
                logger.error(f"Failed to apply optimization {pragma}: {e}")
                continue
        
        conn.commit()
    
    logger.info("Database optimization settings applied")


def run_performance_migration():
    """Run the complete performance optimization migration."""
    logger.info("Starting performance optimization migration...")
    
    try:
        # Create performance indexes
        create_performance_indexes()
        
        # Analyze table statistics
        analyze_table_statistics()
        
        # Apply database optimizations
        optimize_database_settings()
        
        logger.info("Performance optimization migration completed successfully")
        
    except Exception as e:
        logger.error(f"Performance migration failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the migration
    run_performance_migration()