# MomentumML Database Layer

This directory contains the complete database layer for the MomentumML application, implementing SQLite-based persistence for game data, events, and TMI calculations.

## Architecture

The database layer follows a clean architecture pattern with the following components:

- **Models** (`models.py`): SQLAlchemy ORM models
- **Repositories** (`repositories.py`): Data access layer with repository pattern
- **Service** (`service.py`): High-level business logic and operations
- **Migrations** (`migrations.py`): Database schema management and versioning
- **Configuration** (`config.py`): Database connection and session management

## Database Schema

### Tables

1. **games**: Stores basic game information
   - `game_id` (Primary Key): Unique game identifier
   - `home_team`: Home team tricode
   - `away_team`: Away team tricode
   - `game_date`: Game date string
   - `status`: Game status (Live, Final, etc.)
   - `created_at`: Record creation timestamp

2. **events**: Stores play-by-play game events
   - `event_id` (Primary Key): Unique event identifier
   - `game_id` (Foreign Key): Reference to games table
   - `team_tricode`: Team that performed the event
   - `player_name`: Player involved in the event
   - `event_type`: Type of event (shot, rebound, turnover, etc.)
   - `clock`: Game clock time
   - `period`: Game period/quarter
   - `points_total`: Team's total points after event
   - `shot_result`: Made/Missed for shot events
   - `description`: Human-readable event description
   - `timestamp`: Event timestamp

3. **tmi_calculations**: Stores Team Momentum Index calculations
   - `id` (Primary Key): Auto-incrementing ID
   - `game_id` (Foreign Key): Reference to games table
   - `team_tricode`: Team for the TMI calculation
   - `tmi_value`: Calculated TMI value
   - `feature_contributions`: JSON string of feature weights
   - `prediction_probability`: ML prediction probability
   - `confidence_score`: Prediction confidence
   - `rolling_window_size`: Window size used for calculation
   - `calculated_at`: Calculation timestamp

4. **schema_migrations**: Tracks applied database migrations
   - `version`: Migration version identifier
   - `description`: Migration description
   - `applied_at`: Migration application timestamp

### Indexes

Performance indexes are automatically created for:
- Game and event lookups by game_id
- Event queries by team and period
- TMI calculations by game and team
- Time-based queries on calculated_at timestamps

## Usage

### Basic Database Operations

```python
from database.service import DatabaseService
from models.game_models import GameInfo, GameEvent

# Create a database service instance
with DatabaseService() as db_service:
    # Create or update a game
    game_info = GameInfo(
        game_id="game_001",
        home_team="LAL",
        away_team="GSW",
        game_date="2024-01-15",
        status="Live",
        period=1,
        clock="12:00",
        home_score=0,
        away_score=0
    )
    db_service.create_or_update_game(game_info)
    
    # Store events
    events = [...]  # List of GameEvent objects
    stored_count = db_service.store_events(events)
    
    # Retrieve data
    active_games = db_service.get_active_games()
    game_events = db_service.get_game_events("game_001")
```

### FastAPI Integration

```python
from fastapi import Depends
from database.config import get_database
from database.service import DatabaseService

@app.get("/games")
async def get_games(db: Session = Depends(get_database)):
    db_service = DatabaseService(db)
    return db_service.get_active_games()
```

### Repository Pattern

For more granular control, use repositories directly:

```python
from database.config import SessionLocal
from database.repositories import GameRepository, EventRepository

with SessionLocal() as db:
    game_repo = GameRepository(db)
    event_repo = EventRepository(db)
    
    # Direct repository operations
    game = game_repo.get_game("game_001")
    events = event_repo.get_events_by_team("game_001", "LAL")
```

## Database Management

### CLI Commands

The database includes a CLI tool for management operations:

```bash
# Run migrations
python backend/database/cli.py migrate

# Check database health
python backend/database/cli.py health

# Reset database (WARNING: destroys all data)
python backend/database/cli.py reset --confirm

# Create tables
python backend/database/cli.py create

# Drop tables (WARNING: destroys all data)
python backend/database/cli.py drop --confirm
```

### Migrations

The migration system automatically handles schema updates:

```python
from database.migrations import run_migrations

# Run all pending migrations
run_migrations()

# Check database health
from database.migrations import check_database_health
is_healthy = check_database_health()
```

### Environment Configuration

Configure the database using environment variables:

```bash
# Database URL (defaults to SQLite)
DATABASE_URL=sqlite:///./momentum_ml.db

# For PostgreSQL (production)
DATABASE_URL=postgresql://user:password@localhost/momentum_ml
```

## Testing

### Unit Tests

Test individual components:

```bash
# Test basic database operations
python backend/test_database.py

# Test integration with FastAPI
python backend/test_integration.py
```

### Test Data

The test scripts create sample data for verification:
- Test games with realistic team codes
- Sample events with proper timestamps
- TMI calculations with feature contributions

## Performance Considerations

### Indexing Strategy

- Primary keys and foreign keys are automatically indexed
- Composite indexes on frequently queried columns (game_id + timestamp)
- Team-based queries optimized with team_tricode indexes

### Query Optimization

- Repository methods use efficient SQLAlchemy queries
- Batch operations for bulk inserts
- Pagination support for large result sets
- Connection pooling for concurrent access

### Data Cleanup

```python
# Clean up old TMI calculations to prevent bloat
cleanup_stats = db_service.cleanup_old_data("game_001", keep_tmi=100)
```

## Error Handling

The database layer includes comprehensive error handling:

- Connection failures with automatic retry
- Transaction rollback on errors
- Graceful degradation for missing data
- Detailed logging for debugging

## Production Considerations

### Database Choice

- **Development**: SQLite (included)
- **Production**: PostgreSQL recommended
- **Testing**: In-memory SQLite

### Monitoring

- Health check endpoints
- Query performance logging
- Connection pool monitoring
- Migration status tracking

### Backup Strategy

- Regular database backups
- Point-in-time recovery
- Migration rollback capabilities
- Data archival for historical games

## File Structure

```
backend/database/
├── __init__.py
├── README.md           # This file
├── cli.py             # Database management CLI
├── config.py          # Database configuration
├── migrations.py      # Migration system
├── models.py          # SQLAlchemy models
├── repositories.py    # Data access layer
└── service.py         # Business logic layer
```

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **1.1**: Real-time data storage and retrieval
- **1.2**: Game event persistence and processing
- **7.3**: Database query optimization for sub-second response times

The database layer provides a solid foundation for the MomentumML application's data persistence needs.