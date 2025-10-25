# MomentumML

Real-time basketball momentum analytics platform that quantifies and visualizes team momentum throughout NBA games.

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   ├── models/             # Data models and schemas
│   ├── services/           # Business logic services
│   ├── main.py            # FastAPI application entry point
│   └── requirements.txt   # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── public/            # Static assets
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── types/         # TypeScript interfaces
│   │   ├── App.tsx        # Main App component
│   │   └── index.tsx      # Application entry point
│   ├── package.json       # Node.js dependencies
│   └── tsconfig.json      # TypeScript configuration
└── README.md              # Project documentation
```

## Getting Started

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Core Data Models

- **GameEvent**: Individual play-by-play events from NBA games
- **Possession**: Aggregated team possessions with statistics
- **TeamMomentumIndex**: Calculated momentum metrics and predictions

## Next Steps

Follow the implementation plan in `.kiro/specs/momentum-ml/tasks.md` to build out the complete system.