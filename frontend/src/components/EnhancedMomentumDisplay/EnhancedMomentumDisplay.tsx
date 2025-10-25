import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { MomentumData } from '../../types/index';

interface VisualizationMomentumData {
  game_id: string;
  teams: {
    home: string;
    away: string;
  };
  scores: {
    home: number;
    away: number;
  };
  game_status: {
    period: number;
    clock: string;
    status: string;
  };
  momentum_data: {
    momentum_meter: {
      leading_team: string | null;
      strength: string;
      confidence: number;
    };
    team_highlights: {
      [team: string]: {
        has_momentum: boolean;
        momentum_level: string;
        percentage: number;
        recent_score: number;
      };
    };
    momentum_bar: {
      [team: string]: number;
    };
    recent_trend: string | null;
    last_updated: string;
  };
}

interface EnhancedMomentumDisplayProps {
  gameId: string;
  refreshInterval?: number;
  overrideMomentumData?: MomentumData | null; // For time-based analysis
  analysisMode?: boolean; // Whether we're in historical analysis mode
}

const EnhancedMomentumDisplay: React.FC<EnhancedMomentumDisplayProps> = ({
  gameId,
  refreshInterval = 5000,
  overrideMomentumData = null,
  analysisMode = false
}) => {
  const [momentumData, setMomentumData] = useState<VisualizationMomentumData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If we have override data (analysis mode), use that instead of fetching
    if (overrideMomentumData && analysisMode) {
      // Convert MomentumData to the expected format for visualization
      const data = overrideMomentumData as any; // Type assertion to avoid TS issues
      const homeTeam = data.home_team;
      const awayTeam = data.away_team;
      
      const visualizationData = {
        game_id: gameId,
        teams: {
          home: homeTeam.team_tricode,
          away: awayTeam.team_tricode
        },
        scores: {
          home: data.home_score || 0,
          away: data.away_score || 0
        },
        game_status: {
          period: data.quarter || 1,
          clock: typeof data.game_time === 'string' 
            ? data.game_time 
            : `${Math.floor(Number(data.game_time) || 0)}:00`,
          status: "Historical Analysis"
        },
        momentum_data: {
          momentum_meter: {
            leading_team: homeTeam.tmi_value > awayTeam.tmi_value 
              ? homeTeam.team_tricode 
              : awayTeam.team_tricode,
            strength: Math.abs(homeTeam.tmi_value - awayTeam.tmi_value) > 1 
              ? 'strong' : 'moderate',
            confidence: Math.max(homeTeam.confidence_score, awayTeam.confidence_score)
          },
          team_highlights: {
            [homeTeam.team_tricode]: {
              has_momentum: homeTeam.tmi_value > awayTeam.tmi_value,
              momentum_level: homeTeam.tmi_value > 1 ? 'high' : homeTeam.tmi_value > 0 ? 'medium' : 'low',
              percentage: 50 + (homeTeam.tmi_value * 10),
              recent_score: homeTeam.tmi_value
            },
            [awayTeam.team_tricode]: {
              has_momentum: awayTeam.tmi_value > homeTeam.tmi_value,
              momentum_level: awayTeam.tmi_value > 1 ? 'high' : awayTeam.tmi_value > 0 ? 'medium' : 'low',
              percentage: 50 + (awayTeam.tmi_value * 10),
              recent_score: awayTeam.tmi_value
            }
          },
          momentum_bar: {
            [homeTeam.team_tricode]: Math.max(10, Math.min(90, 50 + (homeTeam.tmi_value * 10))),
            [awayTeam.team_tricode]: Math.max(10, Math.min(90, 50 + (awayTeam.tmi_value * 10)))
          },
          recent_trend: homeTeam.tmi_value > awayTeam.tmi_value 
            ? homeTeam.team_tricode 
            : awayTeam.team_tricode,
          last_updated: new Date().toISOString()
        }
      };
      
      setMomentumData(visualizationData);
      setLoading(false);
      setError(null);
      return;
    }

    // Don't fetch if refreshInterval is 0 (analysis mode)
    if (refreshInterval === 0) {
      return;
    }

    const fetchMomentumData = async () => {
      try {
        // Use demo endpoint for demo game, otherwise use regular endpoint
        const endpoint = gameId === 'demo_game_001' 
          ? 'http://localhost:8003/api/games/demo/momentum/visualization'
          : `/api/games/${gameId}/momentum/visualization`;
          
        const response = await fetch(endpoint);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setMomentumData(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch momentum data');
      } finally {
        setLoading(false);
      }
    };

    fetchMomentumData();
    if (refreshInterval > 0) {
      const interval = setInterval(fetchMomentumData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [gameId, refreshInterval, overrideMomentumData, analysisMode]);

  if (loading) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading momentum data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !momentumData) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="text-red-600">
            Error: {error || 'No momentum data available'}
          </div>
        </CardContent>
      </Card>
    );
  }

  const { teams, scores, game_status, momentum_data } = momentumData;
  const { momentum_meter, team_highlights, momentum_bar } = momentum_data;

  // Get team names
  const homeTeam = teams.home;
  const awayTeam = teams.away;

  // Get momentum percentages
  const homePercentage = momentum_bar[homeTeam] || 50;
  const awayPercentage = momentum_bar[awayTeam] || 50;

  // Get team highlights
  const homeHighlight = team_highlights[homeTeam];
  const awayHighlight = team_highlights[awayTeam];

  // Determine which team has momentum
  const leadingTeam = momentum_meter.leading_team;
  const momentumStrength = momentum_meter.strength;
  const confidence = momentum_meter.confidence;

  // Get momentum strength color
  const getMomentumColor = (strength: string) => {
    switch (strength) {
      case 'strong': return 'text-red-600 font-bold';
      case 'moderate': return 'text-orange-500 font-semibold';
      case 'neutral': return 'text-gray-500';
      default: return 'text-gray-400';
    }
  };

  // Get momentum icon
  const getMomentumIcon = (strength: string) => {
    switch (strength) {
      case 'strong': return '🔥';
      case 'moderate': return '⚡';
      case 'neutral': return '⚖️';
      default: return '😴';
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* Enhanced Game Header with Dynamic Momentum Battle */}
      <Card className="w-full overflow-hidden">
        <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-red-50">
          <CardTitle className="text-center text-xl font-bold">
            🏀 MOMENTUM BATTLE
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6 p-6">
          {/* Dynamic Team Cards with Momentum Glow */}
          <div className="relative flex justify-between items-center">
            {/* Home Team Card */}
            <div className={`relative p-4 rounded-xl transition-all duration-500 transform ${
              homeHighlight?.has_momentum 
                ? 'bg-gradient-to-br from-red-100 to-red-200 border-3 border-red-500 text-red-800 shadow-2xl scale-105 animate-pulse' 
                : 'bg-gradient-to-br from-gray-100 to-gray-200 border-2 border-gray-300 text-gray-700 hover:shadow-lg'
            }`}>
              <div className="text-center">
                <div className="text-2xl font-bold mb-1">
                  {homeHighlight?.has_momentum && (
                    <span className="text-3xl mr-2 animate-bounce">{getMomentumIcon(momentumStrength)}</span>
                  )}
                  {homeTeam}
                </div>
                <div className="text-3xl font-black">{scores.home}</div>
                {homeHighlight?.has_momentum && (
                  <div className="text-sm font-semibold mt-1 text-red-600">
                    HAS MOMENTUM!
                  </div>
                )}
              </div>
              {/* Momentum Glow Effect */}
              {homeHighlight?.has_momentum && (
                <div className="absolute inset-0 rounded-xl bg-red-400 opacity-20 animate-pulse"></div>
              )}
            </div>

            {/* VS Indicator with Game Status */}
            <div className="text-center px-4">
              <div className="text-2xl font-bold text-gray-600 mb-1">VS</div>
              <div className="text-sm text-gray-500">
                <div className="font-semibold">Q{game_status.period}</div>
                <div>{game_status.clock}</div>
              </div>
            </div>

            {/* Away Team Card */}
            <div className={`relative p-4 rounded-xl transition-all duration-500 transform ${
              awayHighlight?.has_momentum 
                ? 'bg-gradient-to-br from-red-100 to-red-200 border-3 border-red-500 text-red-800 shadow-2xl scale-105 animate-pulse' 
                : 'bg-gradient-to-br from-gray-100 to-gray-200 border-2 border-gray-300 text-gray-700 hover:shadow-lg'
            }`}>
              <div className="text-center">
                <div className="text-2xl font-bold mb-1">
                  {awayHighlight?.has_momentum && (
                    <span className="text-3xl mr-2 animate-bounce">{getMomentumIcon(momentumStrength)}</span>
                  )}
                  {awayTeam}
                </div>
                <div className="text-3xl font-black">{scores.away}</div>
                {awayHighlight?.has_momentum && (
                  <div className="text-sm font-semibold mt-1 text-red-600">
                    HAS MOMENTUM!
                  </div>
                )}
              </div>
              {/* Momentum Glow Effect */}
              {awayHighlight?.has_momentum && (
                <div className="absolute inset-0 rounded-xl bg-red-400 opacity-20 animate-pulse"></div>
              )}
            </div>
          </div>

          {/* Game Momentum Status Banner */}
          <div className={`text-center p-4 rounded-xl transition-all duration-500 ${
            leadingTeam 
              ? 'bg-gradient-to-r from-orange-100 to-red-100 border-2 border-orange-400' 
              : 'bg-gradient-to-r from-gray-100 to-gray-200 border-2 border-gray-300'
          }`}>
            <div className={`text-xl font-bold ${getMomentumColor(momentumStrength)}`}>
              {leadingTeam ? (
                <>
                  <span className="text-2xl mr-2">{getMomentumIcon(momentumStrength)}</span>
                  <strong className="text-2xl">{leadingTeam}</strong> 
                  <span className="ml-2">controls the game with</span>
                  <span className="ml-2 uppercase tracking-wide">{momentumStrength}</span>
                  <span className="ml-2">momentum</span>
                </>
              ) : (
                <>
                  <span className="text-2xl mr-2">{getMomentumIcon('neutral')}</span>
                  <span className="text-xl">Game momentum is</span>
                  <strong className="ml-2 text-xl uppercase tracking-wide">BALANCED</strong>
                </>
              )}
            </div>
            <div className="text-sm text-gray-600 mt-2 font-semibold">
              AI Confidence: {(confidence * 100).toFixed(0)}% • Model: Improved V2
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Momentum Meter */}
      <Card className="w-full">
        <CardHeader className="pb-2 bg-gradient-to-r from-blue-50 to-red-50">
          <CardTitle className="text-center text-lg font-bold">⚖️ Momentum Distribution</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="space-y-4">
            {/* Team Labels */}
            <div className="flex justify-between items-center mb-2">
              <div className={`text-lg font-bold ${homeHighlight?.has_momentum ? 'text-blue-700' : 'text-blue-500'}`}>
                {homeTeam}
              </div>
              <div className="text-sm text-gray-500 font-semibold">MOMENTUM CONTROL</div>
              <div className={`text-lg font-bold ${awayHighlight?.has_momentum ? 'text-red-700' : 'text-red-500'}`}>
                {awayTeam}
              </div>
            </div>

            {/* Enhanced Visual Momentum Bar */}
            <div className="relative">
              <div className="flex h-12 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                <div 
                  className={`transition-all duration-1000 flex items-center justify-center text-white text-sm font-bold ${
                    homeHighlight?.has_momentum 
                      ? 'bg-gradient-to-r from-blue-600 to-blue-400 shadow-lg' 
                      : 'bg-gradient-to-r from-blue-500 to-blue-300'
                  }`}
                  style={{ width: `${homePercentage}%` }}
                >
                  {homePercentage > 25 && (
                    <span className="flex items-center">
                      {homeHighlight?.has_momentum && '🔥'} {homeTeam}
                    </span>
                  )}
                </div>
                <div 
                  className={`transition-all duration-1000 flex items-center justify-center text-white text-sm font-bold ${
                    awayHighlight?.has_momentum 
                      ? 'bg-gradient-to-l from-red-600 to-red-400 shadow-lg' 
                      : 'bg-gradient-to-l from-red-500 to-red-300'
                  }`}
                  style={{ width: `${awayPercentage}%` }}
                >
                  {awayPercentage > 25 && (
                    <span className="flex items-center">
                      {awayHighlight?.has_momentum && '🔥'} {awayTeam}
                    </span>
                  )}
                </div>
              </div>
              
              {/* Center line with indicator */}
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 h-12 w-1 bg-white shadow-md rounded-full"></div>
              
              {/* Momentum indicator arrow */}
              {leadingTeam && (
                <div className={`absolute top-14 transition-all duration-500 transform -translate-x-1/2 ${
                  homeHighlight?.has_momentum ? 'left-1/4' : 'left-3/4'
                }`}>
                  <div className="text-2xl">
                    {homeHighlight?.has_momentum ? '⬅️' : '➡️'}
                  </div>
                </div>
              )}
            </div>

            {/* Detailed Percentage Breakdown */}
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className={`text-center p-3 rounded-lg ${
                homeHighlight?.has_momentum 
                  ? 'bg-blue-100 border-2 border-blue-400' 
                  : 'bg-gray-100 border border-gray-300'
              }`}>
                <div className={`text-2xl font-bold ${homeHighlight?.has_momentum ? 'text-blue-700' : 'text-blue-600'}`}>
                  {homePercentage.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">{homeTeam} Control</div>
              </div>
              
              <div className={`text-center p-3 rounded-lg ${
                awayHighlight?.has_momentum 
                  ? 'bg-red-100 border-2 border-red-400' 
                  : 'bg-gray-100 border border-gray-300'
              }`}>
                <div className={`text-2xl font-bold ${awayHighlight?.has_momentum ? 'text-red-700' : 'text-red-600'}`}>
                  {awayPercentage.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">{awayTeam} Control</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Team Momentum Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Home Team Detailed Analysis */}
        <Card className={`transition-all duration-500 ${
          homeHighlight?.has_momentum 
            ? 'ring-4 ring-blue-400 shadow-2xl bg-gradient-to-br from-blue-50 to-blue-100' 
            : 'hover:shadow-lg bg-gradient-to-br from-gray-50 to-gray-100'
        }`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-center text-lg font-bold flex items-center justify-center">
              <span className="mr-2">🏠</span>
              {homeTeam} Individual Momentum
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              {/* Momentum Level Badge */}
              <div className={`inline-block px-4 py-2 rounded-full text-lg font-bold mb-3 ${
                homeHighlight?.momentum_level === 'high' 
                  ? 'bg-green-500 text-white shadow-lg' 
                  : homeHighlight?.momentum_level === 'medium' 
                  ? 'bg-yellow-500 text-white shadow-lg' 
                  : 'bg-gray-400 text-white'
              }`}>
                {homeHighlight?.momentum_level?.toUpperCase() || 'LOW'} MOMENTUM
              </div>
              
              {/* Recent Performance Metrics */}
              <div className="space-y-2">
                <div className="flex justify-between items-center p-2 bg-white rounded-lg shadow-sm">
                  <span className="text-sm font-medium text-gray-600">Recent Score:</span>
                  <span className={`font-bold ${
                    (homeHighlight?.recent_score || 0) > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {homeHighlight?.recent_score?.toFixed(1) || '0.0'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-2 bg-white rounded-lg shadow-sm">
                  <span className="text-sm font-medium text-gray-600">Momentum %:</span>
                  <span className="font-bold text-blue-600">
                    {homeHighlight?.percentage?.toFixed(1) || '50.0'}%
                  </span>
                </div>
              </div>
              
              {/* Momentum Status Indicator */}
              {homeHighlight?.has_momentum && (
                <div className="mt-3 p-2 bg-blue-500 text-white rounded-lg font-bold text-sm animate-pulse">
                  🔥 TEAM HAS MOMENTUM ADVANTAGE! 🔥
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Away Team Detailed Analysis */}
        <Card className={`transition-all duration-500 ${
          awayHighlight?.has_momentum 
            ? 'ring-4 ring-red-400 shadow-2xl bg-gradient-to-br from-red-50 to-red-100' 
            : 'hover:shadow-lg bg-gradient-to-br from-gray-50 to-gray-100'
        }`}>
          <CardHeader className="pb-3">
            <CardTitle className="text-center text-lg font-bold flex items-center justify-center">
              <span className="mr-2">✈️</span>
              {awayTeam} Individual Momentum
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              {/* Momentum Level Badge */}
              <div className={`inline-block px-4 py-2 rounded-full text-lg font-bold mb-3 ${
                awayHighlight?.momentum_level === 'high' 
                  ? 'bg-green-500 text-white shadow-lg' 
                  : awayHighlight?.momentum_level === 'medium' 
                  ? 'bg-yellow-500 text-white shadow-lg' 
                  : 'bg-gray-400 text-white'
              }`}>
                {awayHighlight?.momentum_level?.toUpperCase() || 'LOW'} MOMENTUM
              </div>
              
              {/* Recent Performance Metrics */}
              <div className="space-y-2">
                <div className="flex justify-between items-center p-2 bg-white rounded-lg shadow-sm">
                  <span className="text-sm font-medium text-gray-600">Recent Score:</span>
                  <span className={`font-bold ${
                    (awayHighlight?.recent_score || 0) > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {awayHighlight?.recent_score?.toFixed(1) || '0.0'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-2 bg-white rounded-lg shadow-sm">
                  <span className="text-sm font-medium text-gray-600">Momentum %:</span>
                  <span className="font-bold text-red-600">
                    {awayHighlight?.percentage?.toFixed(1) || '50.0'}%
                  </span>
                </div>
              </div>
              
              {/* Momentum Status Indicator */}
              {awayHighlight?.has_momentum && (
                <div className="mt-3 p-2 bg-red-500 text-white rounded-lg font-bold text-sm animate-pulse">
                  🔥 TEAM HAS MOMENTUM ADVANTAGE! 🔥
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Trend Analysis */}
      {momentum_data.recent_trend && (
        <Card className="w-full bg-gradient-to-r from-purple-50 to-indigo-50 border-2 border-purple-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-center text-lg font-bold text-purple-700">
              📈 Recent Momentum Trend
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="text-center">
              <div className="text-lg font-bold mb-2">
                <span className="text-2xl mr-2">🚀</span>
                <span className="text-purple-700">{momentum_data.recent_trend}</span>
                <span className="text-gray-600 ml-2">is trending upward</span>
              </div>
              <div className="text-sm text-gray-600 bg-white p-2 rounded-lg inline-block">
                Based on analysis of the last 20 game events
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Model Information */}
      <Card className="w-full bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200">
        <CardContent className="p-4">
          <div className="text-center space-y-2">
            <div className="text-sm font-semibold text-green-700">
              🤖 Powered by Advanced AI • Model: Improved V2
            </div>
            <div className="text-xs text-gray-600">
              Trained on 2.3M real NBA events • 70% AUC • Real-time predictions
            </div>
            <div className="text-xs text-gray-500">
              Last updated: {new Date(momentum_data.last_updated).toLocaleTimeString()}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EnhancedMomentumDisplay;