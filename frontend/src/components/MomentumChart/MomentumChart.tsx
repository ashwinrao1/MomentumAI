import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { MomentumData } from '../../types';

interface MomentumChartProps {
  momentumHistory: MomentumData[];
  selectedGame: {
    home_team: string;
    away_team: string;
  } | null;
  isLoading?: boolean;
}

const MomentumChart: React.FC<MomentumChartProps> = ({
  momentumHistory,
  selectedGame,
  isLoading = false
}) => {
  const chartData = useMemo(() => {
    if (!momentumHistory.length || !selectedGame) {
      return [];
    }

    console.log('MomentumChart received data:', {
      historyLength: momentumHistory.length,
      firstPoint: momentumHistory[0],
      lastPoint: momentumHistory[momentumHistory.length - 1]
    });

    // Extract time points and TMI values for both teams
    const timePoints = momentumHistory.map((data, index) => {
      // Use numeric game_time (elapsed minutes) for proper line chart
      if (typeof data.game_time === 'number') {
        return data.game_time;
      }
      // Fallback to index if game_time is not numeric
      return index * (48 / momentumHistory.length);
    });

    const homeTeamTMI = momentumHistory.map(data => data.home_team.tmi_value);
    const awayTeamTMI = momentumHistory.map(data => data.away_team.tmi_value);

    // Create hover text with detailed information
    const homeHoverText = momentumHistory.map(data => {
      const features = Object.entries(data.home_team.feature_contributions)
        .map(([key, value]) => `${key}: ${value.toFixed(3)}`)
        .join('<br>');
      
      const timeDisplay = typeof data.game_time === 'number' 
        ? `${Math.floor(data.game_time)}:${Math.floor((data.game_time % 1) * 60).toString().padStart(2, '0')}`
        : data.game_time;
      
      return `<b>${selectedGame.home_team}</b><br>` +
             `TMI: ${data.home_team.tmi_value.toFixed(3)}<br>` +
             `Time: ${timeDisplay}<br>` +
             `Confidence: ${(data.home_team.confidence_score * 100).toFixed(1)}%<br>` +
             `<br><b>Feature Contributions:</b><br>${features}`;
    });

    const awayHoverText = momentumHistory.map(data => {
      const features = Object.entries(data.away_team.feature_contributions)
        .map(([key, value]) => `${key}: ${value.toFixed(3)}`)
        .join('<br>');
      
      const timeDisplay = typeof data.game_time === 'number' 
        ? `${Math.floor(data.game_time)}:${Math.floor((data.game_time % 1) * 60).toString().padStart(2, '0')}`
        : data.game_time;
      
      return `<b>${selectedGame.away_team}</b><br>` +
             `TMI: ${data.away_team.tmi_value.toFixed(3)}<br>` +
             `Time: ${timeDisplay}<br>` +
             `Confidence: ${(data.away_team.confidence_score * 100).toFixed(1)}%<br>` +
             `<br><b>Feature Contributions:</b><br>${features}`;
    });

    return [
      {
        x: timePoints,
        y: homeTeamTMI,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: selectedGame.home_team,
        line: {
          color: '#2563eb', // Blue for home team
          width: 3
        },
        marker: {
          color: '#2563eb',
          size: 6,
          symbol: 'circle'
        },
        hovertemplate: '%{text}<extra></extra>',
        text: homeHoverText,
        connectgaps: true
      },
      {
        x: timePoints,
        y: awayTeamTMI,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: selectedGame.away_team,
        line: {
          color: '#dc2626', // Red for away team
          width: 3
        },
        marker: {
          color: '#dc2626',
          size: 6,
          symbol: 'circle'
        },
        hovertemplate: '%{text}<extra></extra>',
        text: awayHoverText,
        connectgaps: true
      }
    ];
  }, [momentumHistory, selectedGame]);

  const layout = useMemo(() => ({
    title: {
      text: 'Team Momentum Index Over Time',
      font: { size: 18, color: '#374151' },
      x: 0.5
    },
    xaxis: {
      title: {
        text: 'Game Time (Minutes)',
        font: { size: 14, color: '#6b7280' }
      },
      gridcolor: '#e5e7eb',
      tickfont: { size: 12, color: '#6b7280' },
      showgrid: true,
      zeroline: false,
      range: [0, 48],  // Force 0-48 minute range
      tickmode: 'linear' as const,
      tick0: 0,
      dtick: 12  // Show ticks every 12 minutes (quarters)
    },
    yaxis: {
      title: {
        text: 'Team Momentum Index (TMI)',
        font: { size: 14, color: '#6b7280' }
      },
      gridcolor: '#e5e7eb',
      tickfont: { size: 12, color: '#6b7280' },
      showgrid: true,
      zeroline: true,
      zerolinecolor: '#9ca3af',
      zerolinewidth: 2
    },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    font: { family: 'Inter, system-ui, sans-serif' },
    legend: {
      orientation: 'h' as const,
      x: 0.5,
      xanchor: 'center' as const,
      y: -0.15,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#e5e7eb',
      borderwidth: 1,
      font: { size: 12 }
    },
    margin: {
      l: 60,
      r: 40,
      t: 60,
      b: 80
    },
    hovermode: 'closest' as const,
    showlegend: true
  }), []);

  const config = useMemo(() => ({
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'pan2d' as const,
      'select2d' as const,
      'lasso2d' as const,
      'autoScale2d' as const,
      'hoverClosestCartesian' as const,
      'hoverCompareCartesian' as const,
      'toggleSpikelines' as const
    ],
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: {
      format: 'png' as const,
      filename: 'momentum_chart',
      height: 500,
      width: 800,
      scale: 1
    }
  }), []);

  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading momentum data...</p>
        </div>
      </div>
    );
  }

  if (!selectedGame) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-800 mb-2">No Game Selected</h3>
          <p className="text-gray-600">Select a game to view momentum trends</p>
        </div>
      </div>
    );
  }

  if (!momentumHistory.length) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
          </svg>
          <h3 className="text-lg font-medium text-gray-800 mb-2">No Momentum Data</h3>
          <p className="text-gray-600">Waiting for momentum calculations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
        useResizeHandler={true}
      />
    </div>
  );
};

export default MomentumChart;