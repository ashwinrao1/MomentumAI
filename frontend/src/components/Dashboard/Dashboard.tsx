// @ts-nocheck
import React, { useState, useEffect, useCallback } from 'react';
import { GameSelection, MomentumData, WebSocketMessage, UserConfiguration } from '../../types';
import GameSelector from './GameSelector';
import { LazyMomentumChart, LazyFeatureImportance, usePreloadComponents } from '../LazyComponents';
import Scoreboard from '../Scoreboard';
import ErrorDisplay from '../ErrorDisplay';
import ModelStatus from '../ModelStatus';
import { loadUserConfiguration, saveUserConfiguration } from '../../utils/userPreferences';
import { useWebSocket } from '../../hooks/useWebSocket';
import EnhancedMomentumDisplay from '../EnhancedMomentumDisplay/EnhancedMomentumDisplay';
import TimeSelector from '../TimeSelector/TimeSelector';
import {
  AppError,
  createAppError,
  ErrorCategory,
  ErrorSeverity,
  fetchWithErrorHandling,
  gracefulDegradation,
  useErrorState,
  logError
} from '../../utils/errorHandling';

interface DashboardProps {}

const Dashboard: React.FC<DashboardProps> = () => {
  const [selectedGame, setSelectedGame] = useState<GameSelection | null>(null);
  const [availableGames, setAvailableGames] = useState<GameSelection[]>([]);
  const [momentumData, setMomentumData] = useState<MomentumData | null>(null);
  const [momentumHistory, setMomentumHistory] = useState<MomentumData[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<'home' | 'away'>('home');
  const [configuration, setConfiguration] = useState<UserConfiguration>(loadUserConfiguration());
  const [isConfigOpen, setIsConfigOpen] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // New state for time analysis
  const [selectedGameTime, setSelectedGameTime] = useState<number | null>(null);
  const [isTimeAnalysisMode, setIsTimeAnalysisMode] = useState<boolean>(false);
  const [timeAnalysisMomentum, setTimeAnalysisMomentum] = useState<MomentumData | null>(null);
  
  // Error state management
  const { errorState, setError, clearError, retry } = useErrorState();
  
  // Preload components for better UX
  const { preloadMomentumChart, preloadFeatureImportance } = usePreloadComponents();

  // WebSocket connection with enhanced error handling and optimized message processing
  // Only use WebSocket for live games
  const websocketUrl = selectedGame?.status === 'Live' || selectedGame?.status === 'In Progress' 
    ? 'ws://localhost:8003/live/stream' 
    : '';

  const {
    connectionStatus,
    lastMessage,
    connect: connectWebSocket,
    disconnect: disconnectWebSocket,
    reconnectAttempts,
    error: websocketError,
    isHealthy: isWebSocketHealthy
  } = useWebSocket({
    url: websocketUrl,
    onMessage: useCallback((message: WebSocketMessage) => {
      try {
        switch (message.type) {
          case 'momentum_update':
            // Process momentum update data
              // Parse compressed message format
              const optimizedData = {
                ...message.data,
                // Expand shortened field names if present
                teams: Object.fromEntries(
                  Object.entries(message.data.teams || {}).map(([team, data]: [string, any]) => [
                    team,
                    {
                      team_tricode: data.t || data.team_tricode,
                      tmi_value: data.v || data.tmi_value,
                      prediction_probability: data.p || data.prediction_probability,
                      confidence_score: data.c || data.confidence_score,
                      feature_contributions: data.f || data.feature_contributions
                    }
                  ])
                ),
                // Convert timestamp if compressed
                timestamp: message.data.ts ? new Date(message.data.ts * 1000).toISOString() : message.data.timestamp
              };
              
              setMomentumData(optimizedData);
              
              // Cache successful data
              gracefulDegradation.setFallbackData('latest_momentum', optimizedData);
              
              // Add to history with throttling to prevent excessive updates
              setMomentumHistory(prev => {
                // Only add if significantly different from last entry or enough time has passed
                const lastEntry = prev[prev.length - 1];
                const shouldAdd = !lastEntry || 
                  Date.now() - new Date(lastEntry.home_team.timestamp).getTime() > 5000; // 5 second minimum
                
                if (shouldAdd) {
                  const newHistory = [...prev, optimizedData];
                  return newHistory.slice(-configuration.maxHistoryPoints);
                }
                return prev;
              });
              
            // Clear any existing errors on successful data
            if (errorState.hasError && errorState.error?.category === ErrorCategory.WEBSOCKET) {
              clearError();
            }
            break;
            
          case 'game_state':
            gracefulDegradation.setFallbackData('game_state', message.data);
            break;
            
          case 'error':
            const serverError = createAppError(
              message.data.message || 'Server error occurred',
              ErrorCategory.API,
              ErrorSeverity.MEDIUM,
              {
                details: message.data,
                userMessage: 'The server encountered an error. Data may be temporarily unavailable.'
              }
            );
            setError(serverError);
            break;
            
          default:
            // Ignore unknown message types to reduce console noise
            if (process.env.NODE_ENV === 'development') {
              console.log('Unknown message type:', message.type);
            }
        }
      } catch (err) {
        const parseError = createAppError(
          'Failed to process server message',
          ErrorCategory.DATA_PROCESSING,
          ErrorSeverity.LOW,
          {
            originalError: err as Error,
            details: { messageType: message.type }
          }
        );
        setError(parseError);
      }
    }, [configuration.maxHistoryPoints, errorState, clearError, setError]),
    
    onError: useCallback((error: AppError) => {
      setError(error);
    }, [setError]),
    
    maxReconnectAttempts: 10,
    reconnectInterval: 3000
  });

  // Fetch available games with error handling and caching
  const fetchAvailableGames = useCallback(async () => {
    setIsLoading(true);
    
    try {
      const response = await fetchWithErrorHandling(
        'http://localhost:8003/api/momentum/games',
        {},
        { maxRetries: 2, baseDelay: 1000 }
      );
      
      const games = await response.json();
      setAvailableGames(games);
      
      // Cache successful result
      gracefulDegradation.setFallbackData('available_games', games);
      
      // Clear any existing API errors
      if (errorState.hasError && errorState.error?.category === ErrorCategory.API) {
        clearError();
      }
      
    } catch (error) {
      // Try to use cached data
      const cachedGames = gracefulDegradation.getFallbackData('available_games', 300000); // 5 minutes
      if (cachedGames) {
        setAvailableGames(cachedGames);
        console.warn('Using cached games data due to API error');
      } else {
        const apiError = error instanceof AppError ? error : 
          createAppError(
            'Failed to fetch available games',
            ErrorCategory.API,
            ErrorSeverity.MEDIUM,
            {
              originalError: error as Error,
              userMessage: 'Unable to load game list. Please check your connection and try again.'
            }
          );
        setError(apiError);
      }
    } finally {
      setIsLoading(false);
    }
  }, [errorState, clearError, setError]);

  // Fetch momentum data for selected game
  const fetchGameMomentum = useCallback(async (gameId: string) => {
    try {
      setIsLoading(true);
      
      const response = await fetchWithErrorHandling(
        `http://localhost:8003/api/momentum/current?game_id=${gameId}`,
        {},
        { maxRetries: 2, baseDelay: 1000 }
      );
      
      const momentumData = await response.json();
      
      // Check if we have timeline data
      if (momentumData.momentum_timeline && Object.keys(momentumData.momentum_timeline).length > 0) {
        console.log('Processing timeline data:', momentumData.momentum_timeline);
        
        // Convert timeline data to momentum history format
        const timeline = momentumData.momentum_timeline;
        const teams = Object.keys(timeline);
        
        // Determine home and away teams
        const homeTeam = teams.find(team => team === 'HOME') || teams[0];
        const awayTeam = teams.find(team => team === 'AWAY') || teams[1] || teams[0];
        
        const homeTimeline = timeline[homeTeam] || [];
        const awayTimeline = timeline[awayTeam] || [];
        
        console.log(`Timeline lengths: HOME=${homeTimeline.length}, AWAY=${awayTimeline.length}`);
        
        // Simple approach: use the longer timeline as base
        const maxLength = Math.max(homeTimeline.length, awayTimeline.length);
        const timelineHistory: MomentumData[] = [];
        
        console.log(`Creating timeline with ${maxLength} points`);
        
        for (let i = 0; i < maxLength; i++) {
          const homePoint = homeTimeline[i] || homeTimeline[homeTimeline.length - 1];
          const awayPoint = awayTimeline[i] || awayTimeline[awayTimeline.length - 1];
          
          if (i < 3) {  // Debug first few iterations
            console.log(`Iteration ${i}:`, {
              homePoint: homePoint ? 'exists' : 'null',
              awayPoint: awayPoint ? 'exists' : 'null',
              homePointSample: homePoint ? {
                game_time: homePoint.game_time,
                tmi_value: homePoint.tmi_value
              } : null
            });
          }
          
          if (homePoint && awayPoint) {
            timelineHistory.push({
              home_team: {
                game_id: gameId,
                team_tricode: homeTeam,
                timestamp: new Date().toISOString(),
                tmi_value: homePoint.tmi_value || 0,
                feature_contributions: homePoint.feature_contributions || {},
                rolling_window_size: 5,
                prediction_probability: homePoint.prediction_probability || 0.5,
                confidence_score: homePoint.confidence_score || 0.8
              },
              away_team: {
                game_id: gameId,
                team_tricode: awayTeam,
                timestamp: new Date().toISOString(),
                tmi_value: awayPoint.tmi_value || 0,
                feature_contributions: awayPoint.feature_contributions || {},
                rolling_window_size: 5,
                prediction_probability: awayPoint.prediction_probability || 0.5,
                confidence_score: awayPoint.confidence_score || 0.8
              },
              game_time: homePoint.game_time || awayPoint.game_time || i,
              quarter: homePoint.period || awayPoint.period || 1,
              home_score: momentumData.final_scores?.home_score || 0,
              away_score: momentumData.final_scores?.away_score || 0
            });
          }
        }
        
        // Set the timeline history for the chart
        setMomentumHistory(timelineHistory);
        
        // Set current momentum data (latest point)
        if (timelineHistory.length > 0) {
          setMomentumData(timelineHistory[timelineHistory.length - 1]);
        }
        
        console.log(`Created ${timelineHistory.length} timeline points from HOME:${homeTimeline.length}, AWAY:${awayTimeline.length}`);
        
        if (timelineHistory.length === 0) {
          console.error('Timeline creation failed! Debugging info:', {
            homeTimelineLength: homeTimeline.length,
            awayTimelineLength: awayTimeline.length,
            homeFirstPoint: homeTimeline[0],
            awayFirstPoint: awayTimeline[0]
          });
        } else {
          console.log('Sample timeline data:', timelineHistory.slice(0, 3).map(point => ({
            game_time: point.game_time,
            home_tmi: point.home_team.tmi_value,
            away_tmi: point.away_team.tmi_value
          })));
        }
        
      } else {
        // Fallback to single point data
        const teams = Object.values(momentumData.teams);
        const formattedData = {
          home_team: teams[0] as any,
          away_team: teams[1] as any || teams[0],
          game_time: momentumData.last_updated,
          quarter: 4,
          home_score: momentumData.final_scores?.home_score || 0,
          away_score: momentumData.final_scores?.away_score || 0
        };
        
        setMomentumData(formattedData);
        setMomentumHistory([formattedData]);
        
        console.log(`Loaded single momentum data point for game ${gameId}`);
      }
      
    } catch (error) {
      console.warn(`Could not load momentum data for game ${gameId}:`, error);
      // Don't show error for missing momentum data, just log it
    } finally {
      setIsLoading(false);
    }
  }, [configuration.maxHistoryPoints]);

  // Handle game selection with error handling
  const handleGameSelect = useCallback((game: GameSelection) => {
    try {
      setSelectedGame(game);
      setMomentumData(null);
      setMomentumHistory([]);
      setSelectedTeam('home');
      clearError();
      
      // Preload components when game is selected
      preloadMomentumChart();
      preloadFeatureImportance();
      
      // Fetch momentum data for the selected game
      fetchGameMomentum(game.game_id);
      
      // Handle WebSocket connection based on game type
      disconnectWebSocket();
      
      if (game.status === 'Final' || game.status === 'Historical') {
        // For finished/historical games, don't maintain WebSocket connection
        console.log(`Selected historical game: ${game.away_team} @ ${game.home_team} (${game.status})`);
      } else {
        // For live games, connect to WebSocket after a short delay
        setTimeout(() => {
          connectWebSocket();
        }, 100);
        console.log(`Selected live game: ${game.away_team} @ ${game.home_team} (${game.status})`);
      }
      
    } catch (error) {
      const selectionError = createAppError(
        'Failed to select game',
        ErrorCategory.SYSTEM,
        ErrorSeverity.LOW,
        {
          originalError: error as Error,
          details: { gameId: game.game_id }
        }
      );
      setError(selectionError);
    }
  }, [connectWebSocket, disconnectWebSocket, clearError, setError, preloadMomentumChart, preloadFeatureImportance]);

  // Handle team toggle for feature importance
  const handleTeamToggle = useCallback((team: 'home' | 'away') => {
    setSelectedTeam(team);
  }, []);

  // Handle time-based momentum analysis
  const handleTimeAnalysis = useCallback((gameTime: number) => {
    if (!momentumHistory || momentumHistory.length === 0) {
      console.warn('No momentum history available for time analysis');
      return;
    }

    setSelectedGameTime(gameTime);
    setIsTimeAnalysisMode(true);

    // Find the closest momentum data point to the selected time
    const closestPoint = momentumHistory.reduce((closest, current) => {
      const currentTimeDiff = Math.abs((current.game_time || 0) - gameTime);
      const closestTimeDiff = Math.abs((closest.game_time || 0) - gameTime);
      return currentTimeDiff < closestTimeDiff ? current : closest;
    });

    setTimeAnalysisMomentum(closestPoint);
    console.log(`Time analysis: Selected ${gameTime} minutes, found data at ${closestPoint.game_time} minutes`);
  }, [momentumHistory]);

  // Reset time analysis mode
  const resetTimeAnalysis = useCallback(() => {
    setIsTimeAnalysisMode(false);
    setSelectedGameTime(null);
    setTimeAnalysisMomentum(null);
  }, []);

  // Handle configuration changes
  const handleConfigurationChange = useCallback((newConfig: UserConfiguration) => {
    setConfiguration(newConfig);
    saveUserConfiguration(newConfig);
    
    // Apply configuration changes immediately
    setMomentumHistory(prev => prev.slice(-newConfig.maxHistoryPoints));
  }, []);

  // Toggle configuration modal
  const toggleConfiguration = useCallback(() => {
    setIsConfigOpen(prev => !prev);
  }, []);

  // Initialize component
  useEffect(() => {
    fetchAvailableGames();
    
    // Cleanup on unmount
    return () => {
      disconnectWebSocket();
    };
  }, [fetchAvailableGames, disconnectWebSocket]);

  // Connect WebSocket when game is selected
  useEffect(() => {
    if (selectedGame && connectionStatus === 'disconnected') {
      connectWebSocket();
    }
  }, [selectedGame, connectionStatus, connectWebSocket]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">MomentumML</h1>
              <p className="text-blue-100 mt-1">Real-time Basketball Momentum Analytics</p>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus === 'connected' && isWebSocketHealthy ? 'bg-green-400' : 
                  connectionStatus === 'connecting' ? 'bg-yellow-400' : 
                  connectionStatus === 'error' ? 'bg-red-500' : 'bg-red-400'
                }`}></div>
                <span className="text-sm">
                  {connectionStatus === 'connected' && isWebSocketHealthy ? 'Connected' :
                   connectionStatus === 'connecting' ? 'Connecting...' :
                   connectionStatus === 'error' ? 'Error' :
                   reconnectAttempts > 0 ? `Reconnecting (${reconnectAttempts})` : 'Disconnected'}
                </span>
              </div>
              
              {/* Health indicator */}
              {selectedGame && (
                <div className="text-xs text-blue-100">
                  {isWebSocketHealthy ? '✓ Live' : '⚠ Delayed'}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Game Selection and Model Status */}
        <div className="mb-6 space-y-4">
          <GameSelector
            availableGames={availableGames}
            selectedGame={selectedGame}
            onGameSelect={handleGameSelect}
            onRefresh={fetchAvailableGames}
          />
          
          {/* Model Status */}
          <ModelStatus />
        </div>

        {/* Error Display */}
        {(errorState.hasError || websocketError) && (
          <div className="mb-6">
            <ErrorDisplay
              error={websocketError || errorState.error!}
              onRetry={websocketError ? 
                () => {
                  disconnectWebSocket();
                  setTimeout(connectWebSocket, 1000);
                } :
                errorState.error?.retryable ? 
                  () => retry(async () => {
                    if (errorState.error?.category === ErrorCategory.API) {
                      await fetchAvailableGames();
                    }
                  }) : 
                  undefined
              }
              onDismiss={() => {
                clearError();
                // Note: WebSocket errors are handled by the hook itself
              }}
              showDetails={process.env.NODE_ENV === 'development'}
            />
          </div>
        )}
        
        {/* Loading State */}
        {isLoading && (
          <div className="mb-6 bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded">
            <div className="flex items-center">
              <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Loading...</span>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {selectedGame ? (
          <div className="space-y-6">
            {/* Live Scoreboard */}
            <Scoreboard 
              gameData={isTimeAnalysisMode ? timeAnalysisMomentum : momentumData}
              selectedGame={selectedGame}
              connectionStatus={connectionStatus === 'error' ? 'disconnected' : connectionStatus}
            />

            {/* Time Analysis Selector (for historical games or time-based analysis) */}
            {(selectedGame.status === 'Final' || selectedGame.status === 'Historical' || momentumHistory.length > 0) && (
              <TimeSelector
                onTimeSelect={handleTimeAnalysis}
                onReset={resetTimeAnalysis}
                selectedTime={selectedGameTime}
                isActive={isTimeAnalysisMode}
                gameLength={48}
              />
            )}

            {/* Enhanced Momentum Display */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Main Enhanced Momentum Display */}
              <div className="xl:col-span-2">
                <div className={`transition-all duration-300 ${
                  isTimeAnalysisMode ? 'ring-2 ring-orange-400 bg-orange-50 rounded-lg p-4' : ''
                }`}>
                  {isTimeAnalysisMode && (
                    <div className="mb-4 text-center">
                      <div className="inline-flex items-center px-4 py-2 bg-orange-100 text-orange-800 rounded-full text-sm font-medium">
                        <span className="mr-2">📊</span>
                        Historical Analysis Mode - {selectedGameTime?.toFixed(1)} minutes
                      </div>
                    </div>
                  )}
                  <EnhancedMomentumDisplay 
                    gameId={selectedGame.game_id}
                    refreshInterval={isTimeAnalysisMode ? 0 : 5000} // Don't refresh in analysis mode
                    overrideMomentumData={isTimeAnalysisMode ? timeAnalysisMomentum : null}
                    analysisMode={isTimeAnalysisMode}
                  />
                </div>
              </div>
            </div>

            {/* Momentum Visualization Components */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Momentum Chart */}
              <div className={`bg-white rounded-lg shadow-md p-6 transition-all duration-300 ${
                isTimeAnalysisMode ? 'ring-2 ring-orange-300' : ''
              }`}>
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                  Team Momentum Over Time
                  {isTimeAnalysisMode && (
                    <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full">
                      @ {selectedGameTime?.toFixed(1)}min
                    </span>
                  )}
                </h3>
                <LazyMomentumChart
                  momentumHistory={momentumHistory}
                  selectedGame={selectedGame}
                  isLoading={connectionStatus === 'connecting'}
                  highlightTime={isTimeAnalysisMode ? selectedGameTime : undefined}
                />
              </div>

              {/* Feature Importance */}
              <div className={`bg-white rounded-lg shadow-md p-6 transition-all duration-300 ${
                isTimeAnalysisMode ? 'ring-2 ring-orange-300' : ''
              }`}>
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                  Momentum Drivers
                  {isTimeAnalysisMode && (
                    <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full">
                      Historical
                    </span>
                  )}
                </h3>
                <LazyFeatureImportance
                  currentMomentum={isTimeAnalysisMode ? timeAnalysisMomentum : momentumData}
                  selectedTeam={selectedTeam}
                  onTeamToggle={handleTeamToggle}
                  isLoading={connectionStatus === 'connecting'}
                />
              </div>
            </div>

            {/* Current Momentum Data Display */}
            {(isTimeAnalysisMode ? timeAnalysisMomentum : momentumData) && (
              <div className={`bg-white rounded-lg shadow-md p-6 transition-all duration-300 ${
                isTimeAnalysisMode ? 'ring-2 ring-orange-300 bg-orange-50' : ''
              }`}>
                <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                  {isTimeAnalysisMode ? 'Historical' : 'Current'} Momentum Index
                  {isTimeAnalysisMode && (
                    <span className="ml-2 px-3 py-1 bg-orange-200 text-orange-800 text-sm rounded-full">
                      {selectedGameTime?.toFixed(1)} minutes
                    </span>
                  )}
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Home Team TMI */}
                  <div className="text-center">
                    <h4 className="font-medium text-gray-700 mb-2">
                      {(isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.home_team.team_tricode}
                    </h4>
                    <div className="text-3xl font-bold text-blue-600">
                      {(isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.home_team.tmi_value.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      Confidence: {((isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.home_team.confidence_score * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Away Team TMI */}
                  <div className="text-center">
                    <h4 className="font-medium text-gray-700 mb-2">
                      {(isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.away_team.team_tricode}
                    </h4>
                    <div className="text-3xl font-bold text-red-600">
                      {(isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.away_team.tmi_value.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      Confidence: {((isTimeAnalysisMode ? timeAnalysisMomentum : momentumData)?.away_team.confidence_score * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                
                {isTimeAnalysisMode && (
                  <div className="mt-4 text-center">
                    <div className="inline-flex items-center px-4 py-2 bg-orange-100 text-orange-800 rounded-lg text-sm">
                      <span className="mr-2">📊</span>
                      Analyzing momentum at {selectedGameTime?.toFixed(1)} minutes into the game
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : availableGames.length === 0 ? (
          /* No Games Available */
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">No NBA Games Available</h3>
            <p className="text-gray-600 mb-4">
              No current NBA games found for this week. This could be due to:
            </p>
            <ul className="text-sm text-gray-500 text-left max-w-md mx-auto space-y-1">
              <li>• NBA off-season or break period</li>
              <li>• API connectivity issues</li>
              <li>• No games scheduled for today</li>
            </ul>
            <button
              onClick={fetchAvailableGames}
              className="mt-6 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Refresh Games
            </button>
          </div>
        ) : (
          /* No Game Selected */
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Select a Real NBA Game</h3>
            <p className="text-gray-600">Choose a game from the dropdown above to start analyzing momentum data from real NBA games.</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;