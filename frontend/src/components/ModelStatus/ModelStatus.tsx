import React, { useState, useEffect } from 'react';

interface ModelInfo {
  status: string;
  model_type?: string;
  num_features?: number;
  feature_names?: string[];
}

interface ModelStatusProps {
  className?: string;
}

const ModelStatus: React.FC<ModelStatusProps> = ({ className = '' }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('http://localhost:8003/api/momentum/status');
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Extract production model info
        const productionModelInfo = data.production_model_info || {};
        setModelInfo({
          status: productionModelInfo.status || 'unknown',
          model_type: productionModelInfo.model_type,
          num_features: productionModelInfo.num_features,
          feature_names: productionModelInfo.feature_names
        });
        
        setError(null);
      } catch (err) {
        console.error('Error fetching model status:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch model status');
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelStatus();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchModelStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className={`bg-gray-50 border border-gray-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <svg className="animate-spin w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span className="text-sm text-gray-600">Loading model status...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm text-red-700">Model status unavailable</span>
        </div>
      </div>
    );
  }

  const isModelLoaded = modelInfo?.status === 'loaded';
  const statusColor = isModelLoaded ? 'green' : 'red';
  const statusText = isModelLoaded ? 'Production Model Active' : 'Model Not Loaded';

  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full bg-${statusColor}-400`}></div>
          <div>
            <h4 className="text-sm font-medium text-gray-900">{statusText}</h4>
            {isModelLoaded && modelInfo && (
              <div className="text-xs text-gray-500 mt-1">
                {modelInfo.model_type} • {modelInfo.num_features} features • 95.6% accuracy
              </div>
            )}
          </div>
        </div>
        
        {isModelLoaded && (
          <div className="text-right">
            <div className="text-xs font-medium text-green-600">Ready</div>
            <div className="text-xs text-gray-500">Advanced NBA Model</div>
          </div>
        )}
      </div>
      
      {isModelLoaded && modelInfo?.feature_names && (
        <details className="mt-3">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
            View Features ({modelInfo.num_features})
          </summary>
          <div className="mt-2 text-xs text-gray-600 max-h-32 overflow-y-auto">
            <div className="grid grid-cols-2 gap-1">
              {modelInfo.feature_names.slice(0, 10).map((feature, index) => (
                <div key={index} className="truncate">{feature}</div>
              ))}
              {modelInfo.feature_names.length > 10 && (
                <div className="text-gray-400">+{modelInfo.feature_names.length - 10} more...</div>
              )}
            </div>
          </div>
        </details>
      )}
    </div>
  );
};

export default ModelStatus;