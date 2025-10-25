/**
 * Lazy-loaded components for performance optimization.
 * 
 * This module provides lazy loading for heavy components to improve
 * initial page load performance and reduce bundle size.
 */

import React, { Suspense, lazy } from 'react';

// Lazy load heavy components
const MomentumChart = lazy(() => import('./MomentumChart'));
const FeatureImportance = lazy(() => import('./FeatureImportance'));
const ConfigurationControls = lazy(() => import('./ConfigurationControls'));

// Loading fallback component
const ComponentLoader: React.FC<{ height?: string }> = ({ height = "200px" }) => (
  <div 
    className="flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
    style={{ height }}
  >
    <div className="text-center">
      <svg 
        className="animate-spin w-8 h-8 text-gray-400 mx-auto mb-2" 
        fill="none" 
        stroke="currentColor" 
        viewBox="0 0 24 24"
      >
        <path 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          strokeWidth={2} 
          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" 
        />
      </svg>
      <p className="text-sm text-gray-500">Loading component...</p>
    </div>
  </div>
);

// Error boundary for lazy components
class LazyComponentErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Lazy component error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800 text-sm">
            Failed to load component. Please refresh the page.
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}

// Wrapper components with lazy loading and error boundaries
export const LazyMomentumChart: React.FC<any> = (props) => (
  <LazyComponentErrorBoundary>
    <Suspense fallback={<ComponentLoader height="400px" />}>
      <MomentumChart {...props} />
    </Suspense>
  </LazyComponentErrorBoundary>
);

export const LazyFeatureImportance: React.FC<any> = (props) => (
  <LazyComponentErrorBoundary>
    <Suspense fallback={<ComponentLoader height="300px" />}>
      <FeatureImportance {...props} />
    </Suspense>
  </LazyComponentErrorBoundary>
);

export const LazyConfigurationControls: React.FC<any> = (props) => (
  <LazyComponentErrorBoundary>
    <Suspense fallback={<ComponentLoader height="150px" />}>
      <ConfigurationControls {...props} />
    </Suspense>
  </LazyComponentErrorBoundary>
);

// Hook for preloading components when needed
export const usePreloadComponents = () => {
  const preloadMomentumChart = React.useCallback(() => {
    import('./MomentumChart');
  }, []);

  const preloadFeatureImportance = React.useCallback(() => {
    import('./FeatureImportance');
  }, []);

  const preloadConfigurationControls = React.useCallback(() => {
    import('./ConfigurationControls');
  }, []);

  return {
    preloadMomentumChart,
    preloadFeatureImportance,
    preloadConfigurationControls
  };
};

// Performance monitoring hook for lazy components
export const useLazyComponentMetrics = () => {
  const [loadTimes, setLoadTimes] = React.useState<Record<string, number>>({});

  const recordLoadTime = React.useCallback((componentName: string, startTime: number) => {
    const loadTime = performance.now() - startTime;
    setLoadTimes(prev => ({
      ...prev,
      [componentName]: loadTime
    }));
    
    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`${componentName} loaded in ${loadTime.toFixed(2)}ms`);
    }
  }, []);

  return { loadTimes, recordLoadTime };
};