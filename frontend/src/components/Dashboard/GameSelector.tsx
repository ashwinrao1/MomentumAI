import React, { useState } from 'react';
import { GameSelection } from '../../types';

interface GameSelectorProps {
  availableGames: GameSelection[];
  selectedGame: GameSelection | null;
  onGameSelect: (game: GameSelection) => void;
  onRefresh: () => void;
}

const GameSelector: React.FC<GameSelectorProps> = ({
  availableGames,
  selectedGame,
  onGameSelect,
  onRefresh
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await onRefresh();
    setIsRefreshing(false);
  };

  const handleGameSelect = (game: GameSelection) => {
    onGameSelect(game);
    setIsOpen(false);
  };

  const formatGameDisplay = (game: GameSelection) => {
    const date = new Date(game.game_date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
    return `${game.away_team} @ ${game.home_team} (${date})`;
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'live':
      case 'in progress':
        return 'text-green-600 bg-green-100';
      case 'final':
      case 'completed':
        return 'text-gray-600 bg-gray-100';
      case 'scheduled':
      case 'upcoming':
        return 'text-blue-600 bg-blue-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Game Selection</h2>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center space-x-2 px-3 py-1 text-sm text-blue-600 hover:text-blue-800 disabled:opacity-50"
        >
          <svg 
            className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>

      <div className="relative">
        {/* Dropdown Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between px-4 py-3 text-left bg-gray-50 border border-gray-300 rounded-lg hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <div className="flex-1">
            {selectedGame ? (
              <div>
                <div className="font-medium text-gray-900">
                  {formatGameDisplay(selectedGame)}
                </div>
                <div className="flex items-center mt-1">
                  <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(selectedGame.status)}`}>
                    {selectedGame.status}
                  </span>
                </div>
              </div>
            ) : (
              <span className="text-gray-500">Select a game...</span>
            )}
          </div>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-64 overflow-y-auto">
            {availableGames.length === 0 ? (
              <div className="px-4 py-3 text-gray-500 text-center">
                No games available
              </div>
            ) : (
              availableGames.map((game) => (
                <button
                  key={game.game_id}
                  onClick={() => handleGameSelect(game)}
                  className={`w-full px-4 py-3 text-left hover:bg-gray-50 focus:outline-none focus:bg-gray-50 border-b border-gray-100 last:border-b-0 ${
                    selectedGame?.game_id === game.game_id ? 'bg-blue-50' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">
                        {formatGameDisplay(game)}
                      </div>
                      <div className="flex items-center mt-1 space-x-2">
                        <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(game.status)}`}>
                          {game.status}
                        </span>
                        <span className="text-xs text-gray-500">
                          ID: {game.game_id}
                        </span>
                      </div>
                    </div>
                    {selectedGame?.game_id === game.game_id && (
                      <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* Game Count Info */}
      {availableGames.length > 0 && (
        <div className="mt-3 text-sm text-gray-500">
          {availableGames.length} game{availableGames.length !== 1 ? 's' : ''} available
        </div>
      )}
    </div>
  );
};

export default GameSelector;