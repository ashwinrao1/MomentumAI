import React, { useState, useCallback } from 'react';

interface TimeSelectorProps {
  onTimeSelect: (gameTime: number) => void;
  onReset: () => void;
  selectedTime: number | null;
  isActive: boolean;
  gameLength?: number; // Total game length in minutes (default 48 for NBA)
}

const TimeSelector: React.FC<TimeSelectorProps> = ({
  onTimeSelect,
  onReset,
  selectedTime,
  isActive,
  gameLength = 48
}) => {
  const [inputTime, setInputTime] = useState<string>('');
  const [sliderTime, setSliderTime] = useState<number>(24); // Default to halftime

  const handleSliderChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(event.target.value);
    setSliderTime(time);
    onTimeSelect(time);
  }, [onTimeSelect]);

  const handleInputSubmit = useCallback((event: React.FormEvent) => {
    event.preventDefault();
    const time = parseFloat(inputTime);
    
    if (isNaN(time) || time < 0 || time > gameLength) {
      alert(`Please enter a valid time between 0 and ${gameLength} minutes`);
      return;
    }
    
    setSliderTime(time);
    onTimeSelect(time);
    setInputTime('');
  }, [inputTime, onTimeSelect, gameLength]);

  const formatTime = (minutes: number): string => {
    const totalMinutes = Math.floor(minutes);
    const seconds = Math.floor((minutes - totalMinutes) * 60);
    return `${totalMinutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getQuarter = (minutes: number): string => {
    if (minutes <= 12) return 'Q1';
    if (minutes <= 24) return 'Q2';
    if (minutes <= 36) return 'Q3';
    if (minutes <= 48) return 'Q4';
    return 'OT';
  };

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 transition-all duration-300 ${
      isActive ? 'ring-2 ring-blue-500 bg-blue-50' : ''
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center">
          <span className="mr-2">⏰</span>
          Historical Moment Analysis
        </h3>
        {isActive && (
          <button
            onClick={onReset}
            className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors text-sm"
          >
            Reset to Live
          </button>
        )}
      </div>

      {isActive && selectedTime !== null && (
        <div className="mb-4 p-3 bg-blue-100 rounded-lg border border-blue-200">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-800">
              Analyzing: {formatTime(selectedTime)} ({getQuarter(selectedTime)})
            </div>
            <div className="text-sm text-blue-600 mt-1">
              Historical momentum at {selectedTime.toFixed(1)} minutes into the game
            </div>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {/* Time Slider */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Game Time (Minutes)
          </label>
          <div className="relative">
            <input
              type="range"
              min="0"
              max={gameLength}
              step="0.5"
              value={sliderTime}
              onChange={handleSliderChange}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Start (0:00)</span>
              <span>Halftime ({gameLength/2}:00)</span>
              <span>End ({gameLength}:00)</span>
            </div>
          </div>
          <div className="text-center mt-2">
            <span className="text-sm font-medium text-gray-700">
              {formatTime(sliderTime)} - {getQuarter(sliderTime)}
            </span>
          </div>
        </div>

        {/* Quick Time Buttons */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quick Select
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {[
              { label: 'Q1 End', time: 12 },
              { label: 'Halftime', time: 24 },
              { label: 'Q3 End', time: 36 },
              { label: 'Final', time: 48 }
            ].map(({ label, time }) => (
              <button
                key={label}
                onClick={() => {
                  setSliderTime(time);
                  onTimeSelect(time);
                }}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  Math.abs(sliderTime - time) < 0.1
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Manual Time Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Enter Specific Time (Minutes)
          </label>
          <form onSubmit={handleInputSubmit} className="flex space-x-2">
            <input
              type="number"
              min="0"
              max={gameLength}
              step="0.1"
              value={inputTime}
              onChange={(e) => setInputTime(e.target.value)}
              placeholder={`0 - ${gameLength}`}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Analyze
            </button>
          </form>
        </div>

        {/* Info Box */}
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <div className="text-xs text-gray-600">
            <div className="font-medium mb-1">💡 How it works:</div>
            <ul className="space-y-1">
              <li>• Use the slider or buttons to select any moment in the game</li>
              <li>• See momentum analysis at that specific time</li>
              <li>• Compare different moments to understand momentum shifts</li>
              <li>• Reset to return to live/current analysis</li>
            </ul>
          </div>
        </div>
      </div>

      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          border: none;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
      `}</style>
    </div>
  );
};

export default TimeSelector;