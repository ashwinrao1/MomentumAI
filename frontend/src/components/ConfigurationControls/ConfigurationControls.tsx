import React, { useState } from 'react';
import { UserConfiguration } from '../../types';
import { validateConfiguration, resetConfiguration } from '../../utils/userPreferences';

interface ConfigurationControlsProps {
  configuration: UserConfiguration;
  onConfigurationChange: (config: UserConfiguration) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const ConfigurationControls: React.FC<ConfigurationControlsProps> = ({
  configuration,
  onConfigurationChange,
  isOpen,
  onToggle
}) => {
  const [localConfig, setLocalConfig] = useState<UserConfiguration>(configuration);

  const handleInputChange = (field: keyof UserConfiguration, value: number) => {
    const updatedConfig = { ...localConfig, [field]: value };
    setLocalConfig(updatedConfig);
  };

  const handleApplyChanges = () => {
    const validatedConfig = validateConfiguration(localConfig);
    setLocalConfig(validatedConfig);
    onConfigurationChange(validatedConfig);
  };

  const handleReset = () => {
    const defaultConfig = resetConfiguration();
    setLocalConfig(defaultConfig);
    onConfigurationChange(defaultConfig);
  };

  const handleCancel = () => {
    setLocalConfig(configuration);
    onToggle();
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-800">Configuration Settings</h2>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Configuration Form */}
        <div className="p-6 space-y-6">
          {/* Rolling Window Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Rolling Window Size
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="1"
                max="20"
                value={localConfig.rollingWindowSize}
                onChange={(e) => handleInputChange('rollingWindowSize', parseInt(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-600 w-16 text-right">
                {localConfig.rollingWindowSize} possessions
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Number of recent possessions used for momentum calculation
            </p>
          </div>

          {/* Refresh Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data Refresh Rate
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="5"
                max="120"
                step="5"
                value={localConfig.refreshRate}
                onChange={(e) => handleInputChange('refreshRate', parseInt(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-600 w-16 text-right">
                {localConfig.refreshRate}s
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              How often to fetch new data from the NBA API
            </p>
          </div>

          {/* Prediction Confidence Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prediction Confidence Threshold
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={localConfig.predictionConfidenceThreshold}
                onChange={(e) => handleInputChange('predictionConfidenceThreshold', parseFloat(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-600 w-16 text-right">
                {(localConfig.predictionConfidenceThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Minimum confidence level for displaying momentum predictions
            </p>
          </div>

          {/* Chart Update Interval */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chart Update Interval
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="500"
                max="5000"
                step="250"
                value={localConfig.chartUpdateInterval}
                onChange={(e) => handleInputChange('chartUpdateInterval', parseInt(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-600 w-16 text-right">
                {(localConfig.chartUpdateInterval / 1000).toFixed(1)}s
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              How often to update chart visualizations
            </p>
          </div>

          {/* Max History Points */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chart History Points
            </label>
            <div className="flex items-center space-x-3">
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={localConfig.maxHistoryPoints}
                onChange={(e) => handleInputChange('maxHistoryPoints', parseInt(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-600 w-16 text-right">
                {localConfig.maxHistoryPoints} points
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Maximum number of data points to keep in chart history
            </p>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
          <button
            onClick={handleReset}
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
          >
            Reset to Defaults
          </button>
          
          <div className="flex space-x-3">
            <button
              onClick={handleCancel}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleApplyChanges}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              Apply Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfigurationControls;