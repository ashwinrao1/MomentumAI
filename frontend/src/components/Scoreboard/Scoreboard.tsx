import React from 'react';
import { MomentumData, GameSelection } from '../../types';

interface ScoreboardProps {
  gameData: MomentumData | null;
  selectedGame: GameSelection | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
}

const Scoreboard: React.FC<ScoreboardProps> = ({ 
  gameData, 
  selectedGame, 
  connectionStatus 
}) => {
  if (!selectedGame) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-center text-gray-500">
          <p>No game selected</p>
        </div>
      </div>
    );
  }

  // Extract team names from game selection
  const homeTeam = selectedGame.home_team;
  const awayTeam = selectedGame.away_team;

  // Get scores and game state from momentum data
  const homeScore = gameData?.home_score || 0;
  const awayScore = gameData?.away_score || 0;
  const quarter = gameData?.quarter || 1;
  const gameTime = gameData?.game_time || '12:00';
  const gameStatus = selectedGame.status;

  // Determine which team has momentum advantage
  const homeMomentum = gameData?.home_team.tmi_value || 0;
  const awayMomentum = gameData?.away_team.tmi_value || 0;
  const momentumAdvantage = homeMomentum > awayMomentum ? 'home' : 'away';
  const momentumDifference = Math.abs(homeMomentum - awayMomentum);

  // Get prediction probabilities for visual indicators
  const homePrediction = gameData?.home_team.prediction_probability || 0;
  const awayPrediction = gameData?.away_team.prediction_probability || 0;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Game Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="text-sm text-gray-600">
          {new Date(selectedGame.game_date).toLocaleDateString()}
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' : 
            connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
          }`}></div>
          <span className="text-xs text-gray-500 uppercase">{connectionStatus}</span>
        </div>
      </div>

      {/* Main Scoreboard */}
      <div className="grid grid-cols-3 gap-4 items-center">
        {/* Away Team */}
        <div className="text-center">
          <div className={`relative p-4 rounded-lg transition-all duration-300 ${
            momentumAdvantage === 'away' && momentumDifference > 0.5 
              ? 'bg-red-50 border-2 border-red-200' 
              : 'bg-gray-50'
          }`}>
            <div className="text-lg font-bold text-gray-800 mb-1">
              {awayTeam}
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {awayScore}
            </div>
            
            {/* Momentum Prediction Indicator */}
            {gameData && (
              <div className="mt-2">
                <div className="text-xs text-gray-600 mb-1">Momentum</div>
                <div className="flex items-center justify-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${
                    awayPrediction > 0.6 ? 'bg-green-500' :
                    awayPrediction > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-xs font-medium">
                    {(awayPrediction * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Game State */}
        <div className="text-center">
          <div className="bg-gray-800 text-white rounded-lg p-4">
            <div className="text-sm font-medium mb-1">
              {gameStatus === 'Live' ? `Q${quarter}` : gameStatus}
            </div>
            <div className="text-lg font-bold">
              {gameStatus === 'Live' ? gameTime : ''}
            </div>
            {gameStatus === 'Live' && (
              <div className="text-xs text-gray-300 mt-1">
                LIVE
              </div>
            )}
          </div>
          
          {/* Overall Momentum Indicator */}
          {gameData && (
            <div className="mt-3">
              <div className="text-xs text-gray-600 mb-1">Team Momentum</div>
              <div className="flex items-center justify-center">
                <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-500 ${
                      momentumAdvantage === 'home' ? 'bg-blue-500' : 'bg-red-500'
                    }`}
                    style={{ 
                      width: `${Math.min(100, (momentumDifference / 2) * 100)}%`,
                      marginLeft: momentumAdvantage === 'away' ? 'auto' : '0'
                    }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Home Team */}
        <div className="text-center">
          <div className={`relative p-4 rounded-lg transition-all duration-300 ${
            momentumAdvantage === 'home' && momentumDifference > 0.5 
              ? 'bg-blue-50 border-2 border-blue-200' 
              : 'bg-gray-50'
          }`}>
            <div className="text-lg font-bold text-gray-800 mb-1">
              {homeTeam}
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {homeScore}
            </div>
            
            {/* Momentum Prediction Indicator */}
            {gameData && (
              <div className="mt-2">
                <div className="text-xs text-gray-600 mb-1">Momentum</div>
                <div className="flex items-center justify-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${
                    homePrediction > 0.6 ? 'bg-green-500' :
                    homePrediction > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-xs font-medium">
                    {(homePrediction * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Additional Game Metadata */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <span className="font-medium">Status:</span> {gameStatus}
          </div>
          <div>
            <span className="font-medium">Game ID:</span> {selectedGame.game_id}
          </div>
        </div>
        
        {/* TMI Values Display */}
        {gameData && (
          <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
            <div className="text-center">
              <div className="text-gray-600">TMI</div>
              <div className="font-bold text-red-600">
                {gameData.away_team.tmi_value.toFixed(2)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-600">TMI</div>
              <div className="font-bold text-blue-600">
                {gameData.home_team.tmi_value.toFixed(2)}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Scoreboard;