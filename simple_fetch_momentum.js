// Simple momentum fetching function for testing
const fetchGameMomentum = async (gameId) => {
  try {
    const response = await fetch(`http://localhost:8003/api/momentum/current?game_id=${gameId}`);
    const data = await response.json();
    
    console.log('Raw API response:', data);
    
    if (data.momentum_timeline) {
      const timeline = data.momentum_timeline;
      const teams = Object.keys(timeline);
      
      console.log('Timeline teams:', teams);
      console.log('HOME timeline length:', timeline.HOME?.length || 0);
      console.log('AWAY timeline length:', timeline.AWAY?.length || 0);
      
      // Create simple timeline data
      const homeTimeline = timeline.HOME || [];
      const awayTimeline = timeline.AWAY || [];
      
      const timelineHistory = [];
      const maxLength = Math.max(homeTimeline.length, awayTimeline.length);
      
      for (let i = 0; i < maxLength; i++) {
        const homePoint = homeTimeline[i] || homeTimeline[homeTimeline.length - 1];
        const awayPoint = awayTimeline[i] || awayTimeline[awayTimeline.length - 1];
        
        if (homePoint && awayPoint) {
          timelineHistory.push({
            home_team: {
              game_id: gameId,
              team_tricode: 'HOME',
              timestamp: new Date().toISOString(),
              tmi_value: homePoint.tmi_value || 0,
              feature_contributions: homePoint.feature_contributions || {},
              rolling_window_size: 5,
              prediction_probability: homePoint.prediction_probability || 0.5,
              confidence_score: homePoint.confidence_score || 0.8
            },
            away_team: {
              game_id: gameId,
              team_tricode: 'AWAY',
              timestamp: new Date().toISOString(),
              tmi_value: awayPoint.tmi_value || 0,
              feature_contributions: awayPoint.feature_contributions || {},
              rolling_window_size: 5,
              prediction_probability: awayPoint.prediction_probability || 0.5,
              confidence_score: awayPoint.confidence_score || 0.8
            },
            game_time: homePoint.game_time || awayPoint.game_time || '12:00',
            quarter: homePoint.period || awayPoint.period || 1,
            home_score: data.final_scores?.home_score || 0,
            away_score: data.final_scores?.away_score || 0
          });
        }
      }
      
      console.log('Created timeline history:', timelineHistory.length, 'points');
      console.log('Sample points:', timelineHistory.slice(0, 3).map(p => ({
        time: p.game_time,
        home_tmi: p.home_team.tmi_value,
        away_tmi: p.away_team.tmi_value
      })));
      
      return timelineHistory;
    }
    
    return [];
  } catch (error) {
    console.error('Error fetching momentum:', error);
    return [];
  }
};

// Test it
fetchGameMomentum('0022500092').then(result => {
  console.log('Final result:', result.length, 'data points');
});