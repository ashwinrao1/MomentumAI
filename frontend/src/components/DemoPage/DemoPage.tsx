import React from 'react';
import EnhancedMomentumDisplay from '../EnhancedMomentumDisplay/EnhancedMomentumDisplay';

const DemoPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🏀 Enhanced NBA Momentum Visualization
          </h1>
          <p className="text-xl text-gray-600 mb-2">
            Powered by Advanced AI • Real NBA Data • Live Predictions
          </p>
          <p className="text-lg text-gray-500">
            Featuring game-level momentum analysis with team highlighting
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">
            🚀 New Features Showcase
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg border-2 border-blue-200">
              <div className="text-3xl mb-2">🎯</div>
              <h3 className="font-bold text-blue-800">Game-Level Momentum</h3>
              <p className="text-sm text-blue-600">Determines which team controls the game</p>
            </div>
            
            <div className="text-center p-4 bg-red-50 rounded-lg border-2 border-red-200">
              <div className="text-3xl mb-2">🔥</div>
              <h3 className="font-bold text-red-800">Team Highlighting</h3>
              <p className="text-sm text-red-600">Visual emphasis on momentum leaders</p>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg border-2 border-green-200">
              <div className="text-3xl mb-2">🤖</div>
              <h3 className="font-bold text-green-800">Advanced AI Model</h3>
              <p className="text-sm text-green-600">Trained on 2.3M real NBA events</p>
            </div>
          </div>
        </div>

        {/* Enhanced Momentum Display */}
        <EnhancedMomentumDisplay 
          gameId="demo_game_001" 
          refreshInterval={10000} // 10 seconds for demo
        />

        <div className="mt-8 text-center">
          <div className="bg-gradient-to-r from-purple-100 to-indigo-100 rounded-lg p-6 border-2 border-purple-200">
            <h3 className="text-xl font-bold text-purple-800 mb-2">
              🎉 What's New in This Version?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
              <div>
                <h4 className="font-semibold text-purple-700 mb-2">Enhanced Visualization:</h4>
                <ul className="text-sm text-purple-600 space-y-1">
                  <li>• Dynamic team highlighting with glow effects</li>
                  <li>• Animated momentum indicators</li>
                  <li>• Game-level momentum determination</li>
                  <li>• Enhanced momentum meter with percentages</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-purple-700 mb-2">Advanced AI Features:</h4>
                <ul className="text-sm text-purple-600 space-y-1">
                  <li>• Improved V2 model with 70% AUC</li>
                  <li>• Real NBA data training (2.3M events)</li>
                  <li>• Confidence scoring and trend analysis</li>
                  <li>• Individual + combined momentum analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;