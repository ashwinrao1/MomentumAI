# MomentumML API Endpoints

This document describes the REST API endpoints and WebSocket connections available in the MomentumML backend.

## Momentum Endpoints

### GET `/api/momentum/current`

Get current Team Momentum Index (TMI) values for a game.

**Parameters:**
- `game_id` (required): NBA game ID
- `team_tricode` (optional): Specific team code to filter results

**Response:**
```json
{
  "game_id": "string",
  "timestamp": "2023-10-23T17:30:00Z",
  "teams": {
    "LAL": {
      "game_id": "string",
      "team_tricode": "LAL",
      "timestamp": "2023-10-23T17:30:00Z",
      "tmi_value": 0.75,
      "feature_contributions": {
        "points_scored": 0.3,
        "fg_percentage": 0.2,
        "rebounds": 0.15,
        "turnovers": -0.1,
        "fouls": -0.05
      },
      "rolling_window_size": 5,
      "prediction_probability": 0.68,
      "confidence_score": 0.82
    }
  },
  "game_status": "Live",
  "last_updated": "2023-10-23T17:30:00Z"
}
```

### GET `/api/momentum/predict`

Get momentum continuation prediction for a specific team.

**Parameters:**
- `game_id` (required): NBA game ID
- `team_tricode` (required): Team code for prediction

**Response:**
```json
{
  "game_id": "string",
  "team_tricode": "LAL",
  "prediction_probability": 0.68,
  "confidence_score": 0.82,
  "prediction_class": "continue",
  "current_tmi": 0.75,
  "feature_contributions": {
    "points_scored": 0.3,
    "fg_percentage": 0.2
  },
  "timestamp": "2023-10-23T17:30:00Z"
}
```

### GET `/api/momentum/games`

Get list of available games for momentum analysis.

**Response:**
```json
[
  {
    "game_id": "string",
    "home_team": "LAL",
    "away_team": "GSW",
    "game_date": "2023-10-23",
    "status": "Live",
    "home_score": 95,
    "away_score": 88
  }
]
```

### GET `/api/momentum/status`

Get the current status of the momentum calculation service.

**Response:**
```json
{
  "status": "healthy",
  "momentum_engine": "initialized",
  "ml_model": "trained",
  "ml_model_path": "models/momentum_predictor.pkl",
  "feature_count": 12,
  "active_games_count": 5,
  "timestamp": "2023-10-23T17:30:00Z"
}
```

### POST `/api/momentum/refresh`

Force refresh of momentum data for a specific game.

**Parameters:**
- `game_id` (required): NBA game ID to refresh

**Response:**
```json
{
  "status": "refreshed",
  "game_id": "string",
  "teams": {
    "LAL": {
      "team_tricode": "LAL",
      "tmi_value": 0.75,
      "prediction_probability": 0.68,
      "confidence_score": 0.82,
      "rolling_window_size": 5
    }
  },
  "event_count": 150,
  "possession_count": 45,
  "timestamp": "2023-10-23T17:30:00Z"
}
```

## Game Endpoints

### GET `/api/games/active`

Get list of currently active NBA games.

### GET `/api/games/{game_id}/fetch`

Fetch latest play-by-play data for a specific game.

### POST `/api/games/{game_id}/process`

Process game data and compute features.

### GET `/api/games/{game_id}/status`

Get current status and basic info for a game.

## WebSocket Endpoints

### WebSocket `/ws/momentum`

Real-time momentum updates WebSocket endpoint.

**Client Messages:**
```json
// Subscribe to game updates
{
  "action": "subscribe",
  "game_id": "game_id_here"
}

// Unsubscribe from game updates
{
  "action": "unsubscribe",
  "game_id": "game_id_here"
}

// Ping for connection health
{
  "action": "ping"
}
```

**Server Messages:**
```json
// Momentum update
{
  "type": "momentum_update",
  "game_id": "string",
  "timestamp": "2023-10-23T17:30:00Z",
  "game_status": "Live",
  "teams": {
    "LAL": {
      "team_tricode": "LAL",
      "tmi_value": 0.75,
      "feature_contributions": {...},
      "prediction_probability": 0.68,
      "confidence_score": 0.82,
      "rolling_window_size": 5
    }
  },
  "event_count": 150
}

// Connection status
{
  "type": "connected",
  "message": "Connected to MomentumML WebSocket",
  "timestamp": "2023-10-23T17:30:00Z"
}

// Subscription confirmation
{
  "type": "subscribed",
  "game_id": "string",
  "message": "Subscribed to game string",
  "timestamp": "2023-10-23T17:30:00Z"
}

// Ping response
{
  "type": "pong",
  "timestamp": "2023-10-23T17:30:00Z"
}

// Error message
{
  "type": "error",
  "message": "Error description"
}
```

### WebSocket `/ws/status`

System status updates WebSocket endpoint.

**Server Messages:**
```json
{
  "type": "status_update",
  "timestamp": "2023-10-23T17:30:00Z",
  "active_connections": 5,
  "game_subscriptions": {
    "game_id_1": 3,
    "game_id_2": 2
  },
  "service_status": "healthy"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (game/team not found)
- `500`: Internal Server Error
- `503`: Service Unavailable (for health checks)

Error responses follow this format:
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

The API respects NBA API rate limits and implements exponential backoff for failed requests. WebSocket connections are limited to 50 concurrent connections per game.

## Authentication

Currently, no authentication is required for API access. This may change in future versions.