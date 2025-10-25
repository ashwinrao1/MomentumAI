import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { MomentumData } from '../../types';

interface FeatureImportanceProps {
  currentMomentum: MomentumData | null;
  selectedTeam: 'home' | 'away';
  onTeamToggle: (team: 'home' | 'away') => void;
  isLoading?: boolean;
}

const FeatureImportance: React.FC<FeatureImportanceProps> = ({
  currentMomentum,
  selectedTeam,
  onTeamToggle,
  isLoading = false
}) => {
  // Feature display names and descriptions
  const featureInfo = useMemo(() => ({
    'field_goal_pct': {
      name: 'Field Goal %',
      description: 'Shooting efficiency over recent possessions'
    },
    'turnovers': {
      name: 'Turnovers',
      description: 'Ball security and possession control'
    },
    'rebounds_diff': {
      name: 'Rebound Diff',
      description: 'Rebounding advantage over opponent'
    },
    'pace': {
      name: 'Pace',
      description: 'Speed of play and tempo control'
    },
    'fouls': {
      name: 'Fouls',
      description: 'Defensive discipline and aggression'
    },
    'points_scored': {
      name: 'Points Scored',
      description: 'Offensive production efficiency'
    },
    'assists': {
      name: 'Assists',
      description: 'Ball movement and team play'
    },
    'steals': {
      name: 'Steals',
      description: 'Defensive pressure and disruption'
    }
  }), []);

  const chartData = useMemo(() => {
    if (!currentMomentum) {
      return [];
    }

    const teamData = selectedTeam === 'home' ? currentMomentum.home_team : currentMomentum.away_team;
    const contributions = teamData.feature_contributions;

    // Sort features by absolute contribution value
    const sortedFeatures = Object.entries(contributions)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 8); // Show top 8 features

    const featureNames = sortedFeatures.map(([key]) => 
      featureInfo[key as keyof typeof featureInfo]?.name || key
    );
    
    const values = sortedFeatures.map(([, value]) => value);
    
    // Create hover text with descriptions
    const hoverText = sortedFeatures.map(([key, value]) => {
      const info = featureInfo[key as keyof typeof featureInfo];
      return `<b>${info?.name || key}</b><br>` +
             `Contribution: ${value.toFixed(3)}<br>` +
             `${info?.description || 'Feature contribution to momentum'}`;
    });

    // Color bars based on positive/negative contribution
    const colors = values.map(value => value >= 0 ? '#10b981' : '#ef4444'); // Green for positive, red for negative

    return [{
      x: values,
      y: featureNames,
      type: 'bar' as const,
      orientation: 'h' as const,
      marker: {
        color: colors,
        line: {
          color: 'rgba(0,0,0,0.1)',
          width: 1
        }
      },
      hovertemplate: '%{text}<extra></extra>',
      text: hoverText,
      name: 'Feature Contribution'
    }];
  }, [currentMomentum, selectedTeam, featureInfo]);

  const layout = useMemo(() => {
    const teamName = currentMomentum 
      ? (selectedTeam === 'home' ? currentMomentum.home_team.team_tricode : currentMomentum.away_team.team_tricode)
      : 'Team';

    return {
      title: {
        text: `Momentum Drivers - ${teamName}`,
        font: { size: 18, color: '#374151' },
        x: 0.5
      },
      xaxis: {
        title: {
          text: 'Contribution to TMI',
          font: { size: 14, color: '#6b7280' }
        },
        gridcolor: '#e5e7eb',
        tickfont: { size: 12, color: '#6b7280' },
        showgrid: true,
        zeroline: true,
        zerolinecolor: '#9ca3af',
        zerolinewidth: 2
      },
      yaxis: {
        title: {
          text: 'Features',
          font: { size: 14, color: '#6b7280' }
        },
        tickfont: { size: 12, color: '#6b7280' },
        showgrid: false,
        automargin: true
      },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      font: { family: 'Inter, system-ui, sans-serif' },
      margin: {
        l: 120,
        r: 40,
        t: 60,
        b: 60
      },
      hovermode: 'closest' as const,
      showlegend: false
    };
  }, [currentMomentum, selectedTeam]);

  const config = useMemo(() => ({
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'pan2d' as const,
      'select2d' as const,
      'lasso2d' as const,
      'autoScale2d' as const,
      'hoverClosestCartesian' as const,
      'hoverCompareCartesian' as const,
      'toggleSpikelines' as const,
      'zoom2d' as const,
      'zoomIn2d' as const,
      'zoomOut2d' as const
    ],
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: {
      format: 'png' as const,
      filename: 'feature_importance',
      height: 400,
      width: 600,
      scale: 1
    }
  }), []);

  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading feature data...</p>
        </div>
      </div>
    );
  }

  if (!currentMomentum) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-center">
          <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="text-lg font-medium text-gray-800 mb-2">No Momentum Data</h3>
          <p className="text-gray-600">Waiting for momentum calculations...</p>
        </div>
      </div>
    );
  }

  const homeTeam = currentMomentum.home_team.team_tricode;
  const awayTeam = currentMomentum.away_team.team_tricode;

  return (
    <div className="w-full">
      {/* Team Toggle Buttons */}
      <div className="flex justify-center mb-4">
        <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-1">
          <button
            onClick={() => onTeamToggle('home')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              selectedTeam === 'home'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            {homeTeam}
          </button>
          <button
            onClick={() => onTeamToggle('away')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              selectedTeam === 'away'
                ? 'bg-red-600 text-white shadow-sm'
                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            {awayTeam}
          </button>
        </div>
      </div>

      {/* Feature Importance Chart */}
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
        useResizeHandler={true}
      />

      {/* Legend */}
      <div className="mt-4 text-sm text-gray-600">
        <div className="flex items-center justify-center space-x-6">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
            <span>Positive Impact</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
            <span>Negative Impact</span>
          </div>
        </div>
        <p className="text-center mt-2 text-xs">
          Features are ranked by their contribution to the current Team Momentum Index
        </p>
      </div>
    </div>
  );
};

export default FeatureImportance;