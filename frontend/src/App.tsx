import React from 'react';
import { Dashboard } from './components/Dashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      {/* Simple Navigation */}
      <nav className="bg-gray-800 text-white p-4">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">🏀 MomentumML</h1>
          <div className="text-sm text-gray-300">
            Real NBA Games • 2025-26 Season
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <Dashboard />
    </div>
  );
}

export default App;