# Design Document

## Overview

MomentumML is a real-time basketball analytics platform that transforms live NBA game data into actionable momentum insights. The system architecture consists of a FastAPI backend that polls the NBA API, processes game events through a momentum calculation engine, and serves real-time updates to a React-based dashboard via WebSocket connections.

The core innovation is the Team Momentum Index (TMI), a composite metric that quantifies team control using rolling statistics over recent possessions. Combined with machine learning predictions, this provides users with both current momentum state and forecasts of likely momentum shifts.

## Architecture

### System Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NBA API       │    │   FastAPI        │    │   React         │
│   (nba_api)     │───▶│   Backend        │───▶│   Dashboard     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          ▲
                              ▼                          │
                       ┌──────────────────┐              │
                       │   SQLite         │              │
                       │   Database       │              │
                       └──────────────────┘              │
                              │                          │
                              ▼                          │
                       ┌──────────────────┐              │
                       │   Momentum       │              │
                       │   Engine         │──────────────┘
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   ML Prediction  │
                       │   Model          │
                       └──────────────────┘
```

### Data Flow

1. **Data Ingestion**: FastAPI backend polls NBA API every 20-30 seconds
2. **Event Processing**: Raw play-by-play events are parsed and stored
3. **Feature Calculation**: Possession-level statistics are computed
4. **TMI Computation**: Rolling momentum index is calculated using weighted features
5. **ML Prediction**: Trained model predicts momentum continuation probability
6. **Real-time Updates**: WebSocket pushes updates to connected dashboard clients

## Components and Interfaces

### Backend Components

#### 1. Live Data Fetcher (`live_fetcher.py`)
- **Purpose**: Polls NBA API and extracts game events
- **Key Methods**:
  - `fetch_live_game_data(game_id)`: Retrieves latest play-by-play
  - `parse_events(raw_data)`: Converts API response to standardized events
  - `get_active_games()`: Lists currently live games

#### 2. Momentum Engine (`momentum_engine.py`)
- **Purpose**: Core TMI calculation and feature engineering
- **Key Methods**:
  - `calculate_possession_features(events)`: Computes per-possession metrics
  - `compute_tmi(features, weights)`: Calculates Team Momentum Index
  - `update_rolling_window(new_events)`: Maintains sliding window of recent possessions

#### 3. ML Prediction Service (`models/predictor.py`)
- **Purpose**: Momentum direction prediction using trained model
- **Key Methods**:
  - `predict_momentum_continuation(current_features)`: Returns probability scores
  - `load_trained_model()`: Initializes pre-trained logistic regression model
  - `extract_prediction_features(tmi_history)`: Prepares input for model inference

#### 4. API Endpoints (`api/main.py`)
- **FastAPI Routes**:
  - `GET /game/{game_id}/fetch`: Pull latest play-by-play data
  - `POST /game/{game_id}/process`: Compute features and update TMI
  - `GET /momentum/current`: Return latest TMI and driver metrics
  - `GET /momentum/predict`: Return next TMI prediction
  - `WebSocket /live/stream`: Stream updates to dashboard

### Frontend Components

#### 1. Dashboard Container (`src/components/Dashboard.jsx`)
- **Purpose**: Main layout and state management
- **Key Features**:
  - WebSocket connection management
  - Game selection interface
  - Real-time data updates

#### 2. Momentum Chart (`src/components/MomentumChart.jsx`)
- **Purpose**: Time-series visualization of TMI
- **Implementation**: Plotly.js line chart with dual y-axes
- **Features**: Interactive tooltips, zoom/pan, team color coding

#### 3. Feature Importance Panel (`src/components/FeatureImportance.jsx`)
- **Purpose**: Bar chart showing momentum drivers
- **Implementation**: Plotly.js horizontal bar chart
- **Features**: Real-time updates, hover explanations

#### 4. Live Scoreboard (`src/components/Scoreboard.jsx`)
- **Purpose**: Current game state display
- **Features**: Score, time remaining, quarter, team logos

## Data Models

### Event Model
```python
@dataclass
class GameEvent:
    event_id: str
    game_id: str
    team_tricode: str
    player_name: str
    event_type: str  # shot, rebound, turnover, foul
    clock: str
    period: int
    points_total: int
    shot_result: Optional[str]  # Made, Missed
    timestamp: datetime
```

### Possession Model
```python
@dataclass
class Possession:
    possession_id: str
    game_id: str
    team_tricode: str
    start_time: str
    end_time: str
    events: List[GameEvent]
    points_scored: int
    fg_attempts: int
    fg_made: int
    turnovers: int
    rebounds: int
    fouls: int
```

### TMI Model
```python
@dataclass
class TeamMomentumIndex:
    game_id: str
    team_tricode: str
    timestamp: datetime
    tmi_value: float
    feature_contributions: Dict[str, float]
    rolling_window_size: int
    prediction_probability: float
    confidence_score: float
```

### Database Schema

```sql
-- Games table
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    game_date DATE NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events table
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    team_tricode TEXT NOT NULL,
    player_name TEXT,
    event_type TEXT NOT NULL,
    clock TEXT NOT NULL,
    period INTEGER NOT NULL,
    points_total INTEGER DEFAULT 0,
    shot_result TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- TMI calculations table
CREATE TABLE tmi_calculations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    team_tricode TEXT NOT NULL,
    tmi_value REAL NOT NULL,
    feature_contributions TEXT, -- JSON string
    prediction_probability REAL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);
```

## Error Handling

### API Error Handling
- **Network Failures**: Implement exponential backoff retry logic (max 3 attempts)
- **Rate Limiting**: Respect NBA API rate limits with appropriate delays
- **Invalid Responses**: Validate API response structure before processing
- **Timeout Handling**: Set 30-second timeout for API requests

### Data Processing Errors
- **Missing Events**: Handle gaps in play-by-play data gracefully
- **Invalid Game States**: Validate game status before processing
- **Feature Calculation Errors**: Use default values for missing statistics
- **Model Prediction Failures**: Fallback to historical averages

### Frontend Error Handling
- **WebSocket Disconnections**: Automatic reconnection with exponential backoff
- **Chart Rendering Errors**: Display error messages and retry options
- **Data Loading States**: Show loading spinners and progress indicators
- **Network Connectivity**: Offline mode with cached data display

### Error Logging
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('momentum_ml.log'),
        logging.StreamHandler()
    ]
)

# Error categories
logger = logging.getLogger('momentum_ml')
logger.error(f"API fetch failed for game {game_id}: {error}")
logger.warning(f"Missing events detected: {missing_count}")
logger.info(f"TMI calculated successfully: {tmi_value}")
```

## Testing Strategy

### Unit Testing
- **Momentum Engine**: Test TMI calculations with known input/output pairs
- **Feature Engineering**: Validate possession-level statistic calculations
- **ML Model**: Test prediction accuracy with historical data
- **API Endpoints**: Mock NBA API responses for consistent testing

### Integration Testing
- **End-to-End Data Flow**: Test complete pipeline from API to dashboard
- **WebSocket Communication**: Verify real-time updates reach frontend
- **Database Operations**: Test data persistence and retrieval
- **Error Recovery**: Simulate failures and verify graceful handling

### Performance Testing
- **Load Testing**: Simulate 50 concurrent users accessing dashboard
- **API Response Times**: Ensure sub-2-second response times
- **Memory Usage**: Monitor memory consumption during extended operation
- **Database Query Performance**: Optimize queries for sub-1-second execution

### Test Data Strategy
- **Historical Games**: Use completed games for deterministic testing
- **Mock API Responses**: Create realistic test data for development
- **Edge Cases**: Test with unusual game scenarios (overtime, technical fouls)
- **Performance Benchmarks**: Establish baseline metrics for regression testing

## Configuration Management

### Environment Variables
```bash
# API Configuration
NBA_API_BASE_URL=https://stats.nba.com
API_POLL_INTERVAL=25  # seconds
API_TIMEOUT=30  # seconds

# Database
DATABASE_URL=sqlite:///momentum_ml.db

# ML Model
MODEL_PATH=models/momentum_predictor.pkl
PREDICTION_THRESHOLD=0.6

# WebSocket
WEBSOCKET_HEARTBEAT=30  # seconds
MAX_CONNECTIONS=100

# Feature Engineering
DEFAULT_WINDOW_SIZE=5  # possessions
TMI_WEIGHTS=0.4,0.25,0.15,0.15,0.05  # score,fg%,rebounds,turnovers,fouls
```

### Deployment Configuration
- **Development**: Local SQLite, debug logging enabled
- **Production**: Optimized database connections, error-only logging
- **Staging**: Production-like environment for testing deployments