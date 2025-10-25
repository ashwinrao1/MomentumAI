#!/usr/bin/env python3
"""
Simple test backend for MomentumML to debug connection issues.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Simple MomentumML Test API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Simple MomentumML API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-10-25T10:00:00Z",
        "version": "1.0.0",
        "environment": "development",
        "database": {"status": "connected", "tables_count": 3},
        "services": {
            "momentum_engine": "operational",
            "ml_predictor": "trained",
            "data_collector": "ready"
        },
        "uptime_seconds": 100
    }

@app.get("/api/momentum/games")
async def get_games():
    return [
        {
            "game_id": "0022400001",
            "home_team": "LAL",
            "away_team": "GSW",
            "game_date": "2025-10-25",
            "status": "Live",
            "home_score": 58,
            "away_score": 62
        },
        {
            "game_id": "0022400002",
            "home_team": "BOS", 
            "away_team": "MIA",
            "game_date": "2025-10-25",
            "status": "Live",
            "home_score": 72,
            "away_score": 68
        },
        {
            "game_id": "0022400003",
            "home_team": "PHX",
            "away_team": "DEN", 
            "game_date": "2025-10-25",
            "status": "Final",
            "home_score": 115,
            "away_score": 108
        }
    ]

@app.get("/api/momentum/status")
async def get_status():
    return {
        "status": "operational",
        "momentum_engine": "initialized",
        "ml_model": "trained",
        "production_ml_model": "loaded",
        "production_model_info": {
            "status": "loaded",
            "model_type": "RandomForestClassifier",
            "num_features": 29,
            "feature_names": [
                "shots", "made_shots", "missed_shots", "rebounds", "turnovers",
                "steals", "blocks", "assists", "fouls", "fg_percentage",
                "shot_attempts_per_event", "points_per_possession", "turnover_rate",
                "steal_rate", "block_rate", "defensive_events", "scoring_run",
                "defensive_run", "shot_clustering", "turnover_clustering",
                "momentum_swings", "avg_period", "late_game", "avg_time_remaining",
                "clutch_time", "shooting_trend", "turnover_trend", "momentum_score",
                "momentum_ratio"
            ]
        },
        "timestamp": "2025-10-25T10:00:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)