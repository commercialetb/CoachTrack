"""
Tactical AI Module
- Tactical Pattern Recognition
- Auto-Scout Report Generator (with Groq)
- Lineup Optimizer (ML-based)
- Game State Analysis
"""

import numpy as np
import pandas as pd
from itertools import combinations
from groq_integration import generate_scout_report_nlg, game_assistant_chat

# =================================================================
# TACTICAL PATTERN RECOGNITION
# =================================================================

class TacticalPatternRecognizer:
    """
    Recognize offensive and defensive patterns from tracking data
    """
    
    def __init__(self):
        self.patterns = {
            'pick_and_roll': {'frequency': 0, 'success_rate': 0},
            'isolation': {'frequency': 0, 'success_rate': 0},
            'motion_offense': {'frequency': 0, 'success_rate': 0},
            'transition': {'frequency': 0, 'success_rate': 0},
            'post_up': {'frequency': 0, 'success_rate': 0}
        }
    
    def analyze_team_patterns(self, team_tracking_data, possession_outcomes):
        """
        Analyze team offensive patterns
        
        Args:
            team_tracking_ dict of player_id -> tracking dataframe
            possession_outcomes: list of possession results (score/turnover/miss)
        """
        
        all_possessions = self._segment_possessions(team_tracking_data)
        
        for i, possession in enumerate(all_possessions):
            pattern = self._classify_possession_pattern(possession)
            outcome = possession_outcomes[i] if i < len(possession_outcomes) else 'miss'
            
            if pattern in self.patterns:
                self.patterns[pattern]['frequency'] += 1
                if outcome == 'score':
                    self.patterns[pattern]['success_rate'] += 1
        
        # Calculate success rates
        for pattern in self.patterns:
            freq = self.patterns[pattern]['frequency']
            if freq > 0:
                self.patterns[pattern]['success_rate'] = round(
                    self.patterns[pattern]['success_rate'] / freq * 100, 1
                )
        
        return self.patterns
    
    def _segment_possessions(self, team_data):
        """Segment tracking data into individual possessions"""
        # Simplified - in real implementation would use time gaps and ball position
        possessions = []
        
        for player_id, data in team_data.items():
            # Split by time gaps > 5 seconds (new possession)
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp')
                time_diff = data['timestamp'].diff()
                possession_breaks = time_diff[time_diff > 5000].index
                
                for i in range(len(possession_breaks) - 1):
                    start_idx = possession_breaks[i]
                    end_idx = possession_breaks[i + 1]
                    possessions.append(data.loc[start_idx:end_idx])
        
        return possessions[:50]  # Return first 50 possessions
    
    def _classify_possession_pattern(self, possession_data):
        """Classify possession into pattern type based on movement"""
        if len(possession_data) < 5:
            return 'transition'
        
        # Calculate movement characteristics
        total_distance = np.sum(np.sqrt(
            np.diff(possession_data['x'].values)**2 + 
            np.diff(possession_data['y'].values)**2
        ))
        
        avg_speed = possession_data.get('speed_kmh_calc', pd.Series([10])).mean()
        
        # Simple heuristic classification
        if avg_speed > 15:
            return 'transition'
        elif total_distance < 10:
            return 'isolation'
        elif total_distance > 50:
            return 'motion_offense'
        elif possession_data['x'].mean() < 10:  # Near basket
            return 'post_up'
        else:
            return 'pick_and_roll'
    
    def get_pattern_summary(self):
        """Get formatted summary of patterns"""
        summary = []
        
        total_possessions = sum(p['frequency'] for p in self.patterns.values())
        
        for pattern, stats in sorted(self.patterns.items(), key=lambda x: x[1]['frequency'], reverse=True):
            if stats['frequency'] > 0:
                pct = stats['frequency'] / total_possessions * 100
                summary.append({
                    'pattern': pattern.replace('_', ' ').title(),
                    'frequency': stats['frequency'],
                    'percentage': round(pct, 1),
                    'success_rate': stats['success_rate']
                })
        
        return summary

# =================================================================
# AUTO-SCOUT REPORT GENERATOR
# =================================================================

class ScoutReportGenerator:
    """
    Generate comprehensive scouting reports using Groq AI
    """
    
    def __init__(self):
        self.opponent_data = {}
    
    def generate_full_report(self, opponent_team, opponent_tracking_data, 
                            opponent_stats, game_videos=None, language='it'):
        """
        Generate complete scouting report
        
        Args:
            opponent_team: str - team name
            opponent_tracking_ dict - tracking data for analysis
            opponent_stats: dict - team statistics
            game_videos: list - video file paths (optional)
            language: str - report language
        """
        
        # Analyze patterns
        pattern_recognizer = TacticalPatternRecognizer()
        
        # Simulate possession outcomes for demo
        n_possessions = 50
        outcomes = np.random.choice(['score', 'miss', 'turnover'], n_possessions, p=[0.45, 0.45, 0.10])
        
        patterns = pattern_recognizer.analyze_team_patterns(opponent_tracking_data, outcomes)
        pattern_summary = pattern_recognizer.get_pattern_summary()
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(opponent_stats, pattern_summary)
        weaknesses = self._identify_weaknesses(opponent_stats, pattern_summary)
        
        # Format data for Groq
        stats_str = self._format_stats_for_groq(opponent_stats)
        patterns_str = self._format_patterns_for_groq(pattern_summary)
        strengths_str = "\n".join([f"- {s}" for s in strengths])
        weaknesses_str = "\n".join([f"- {w}" for w in weaknesses])
        
        # Generate natural language report with Groq
        groq_report = generate_scout_report_nlg(
            opponent_team,
            stats_str,
            patterns_str,
            strengths_str,
            weaknesses_str,
            language
        )
        
        return {
            'team': opponent_team,
            'report_text': groq_report,
            'patterns': pattern_summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'stats': opponent_stats,
            'recommendations': self._generate_game_plan(strengths, weaknesses)
        }
    
    def _identify_strengths(self, stats, patterns):
        """Identify team strengths from stats and patterns"""
        strengths = []
        
        if stats.get('three_pt_pct', 0) > 38:
            strengths.append(f"Ottimo tiro da 3 punti ({stats['three_pt_pct']}%)")
        
        if stats.get('pace', 0) > 100:
            strengths.append(f"Gioco veloce (Pace: {stats['pace']})")
        
        if stats.get('assists_per_game', 0) > 25:
            strengths.append(f"Ottimo movimento palla ({stats['assists_per_game']} assist/game)")
        
        # From patterns
        for pattern in patterns:
            if pattern['success_rate'] > 60:
                strengths.append(f"Efficace in {pattern['pattern']} ({pattern['success_rate']}% successo)")
        
        if not strengths:
            strengths.append("Squadra equilibrata senza punti di forza dominanti")
        
        return strengths
    
    def _identify_weaknesses(self, stats, patterns):
        """Identify team weaknesses"""
        weaknesses = []
        
        if stats.get('turnovers_per_game', 20) > 15:
            weaknesses.append(f"Troppe perse ({stats['turnovers_per_game']}/game)")
        
        if stats.get('def_rating', 115) > 112:
            strengths.append(f"Difesa vulnerabile (Def Rating: {stats['def_rating']})")
        
        if stats.get('ft_pct', 80) < 75:
            weaknesses.append(f"Tiri liberi problematici ({stats['ft_pct']}%)")
        
        if stats.get('rebounds_per_game', 40) < 42:
            weaknesses.append(f"Rimbalzi sotto media ({stats['rebounds_per_game']}/game)")
        
        # From patterns
        for pattern in patterns:
            if pattern['success_rate'] < 40 and pattern['frequency'] > 10:
                weaknesses.append(f"Inefficace in {pattern['pattern']} ({pattern['success_rate']}%)")
        
        if not weaknesses:
            weaknesses.append("Poche debolezze evidenti - squadra solida")
        
        return weaknesses
    
    def _format_stats_for_groq(self, stats):
        """Format stats as string for Groq prompt"""
        return f"""
Punti per partita: {stats.get('points_per_game', 'N/A')}
% Tiro da 3: {stats.get('three_pt_pct', 'N/A')}%
Assist per partita: {stats.get('assists_per_game', 'N/A')}
Perse per partita: {stats.get('turnovers_per_game', 'N/A')}
Pace: {stats.get('pace', 'N/A')}
Offensive Rating: {stats.get('off_rating', 'N/A')}
Defensive Rating: {stats.get('def_rating', 'N/A')}
Rimbalzi/game: {stats.get('rebounds_per_game', 'N/A')}
"""
    
    def _format_patterns_for_groq(self, patterns):
        """Format patterns for Groq"""
        pattern_strs = []
        for p in patterns:
            pattern_strs.append(
                f"- {p['pattern']}: {p['frequency']} volte ({p['percentage']}%), "
                f"successo {p['success_rate']}%"
            )
        return "\n".join(pattern_strs)
    
    def _generate_game_plan(self, strengths, weaknesses):
        """Generate tactical game plan recommendations"""
        recommendations = []
        
        recommendations.append("🎯 **STRATEGIA OFFENSIVA:**")
        if "Difesa vulnerabile" in str(weaknesses):
            recommendations.append("   - Push tempo e cerca canestri facili in transizione")
        if "Rimbalzi" in str(weaknesses):
            recommendations.append("   - Attacca rimbalzo offensivo aggressivamente")
        
        recommendations.append("\n🛡️ **STRATEGIA DIFENSIVA:**")
        if "tiro da 3" in str(strengths):
            recommendations.append("   - Close-out aggressivi sui tiratori, forza penetrazioni")
        if "veloce" in str(strengths):
            recommendations.append("   - Rallenta ritmo, difesa posizionata, evita turnovers")
        
        return recommendations

# =================================================================
# LINEUP OPTIMIZER
# =================================================================

class LineupOptimizer:
    """
    Optimize starting lineup and rotations using ML and statistics
    """
    
    def __init__(self):
        self.player_ratings = {}
        self.chemistry_matrix = None
    
    def calculate_player_ratings(self, player_stats_dict):
        """
        Calculate overall rating for each player
        
        Args:
            player_stats_dict: dict of player_id -> stats dict
        """
        
        for player_id, stats in player_stats_dict.items():
            # Weighted rating calculation
            rating = (
                stats.get('points', 0) * 1.0 +
                stats.get('assists', 0) * 1.5 +
                stats.get('rebounds', 0) * 1.2 +
                stats.get('steals', 0) * 2.0 +
                stats.get('blocks', 0) * 2.0 -
                stats.get('turnovers', 0) * 1.5
            ) / stats.get('minutes', 30) * 10
            
            self.player_ratings[player_id] = round(rating, 2)
        
        return self.player_ratings
    
    def calculate_chemistry(self, team_tracking_data):
        """Calculate player chemistry matrix based on spacing/movement"""
        players = list(team_tracking_data.keys())
        n = len(players)
        chemistry = np.eye(n)  # Initialize with 1s on diagonal
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i < j:
                    # Calculate chemistry based on spacing when on court together
                    chem_score = self._calculate_pair_chemistry(
                        team_tracking_data[player1],
                        team_tracking_data[player2]
                    )
                    chemistry[i, j] = chem_score
                    chemistry[j, i] = chem_score
        
        self.chemistry_matrix = pd.DataFrame(chemistry, index=players, columns=players)
        return self.chemistry_matrix
    
    def _calculate_pair_chemistry(self, player1_data, player2_data):
        """Calculate chemistry between two players"""
        # Simple heuristic: average distance between players
        # Good spacing = higher chemistry
        
        if len(player1_data) < 10 or len(player2_data) < 10:
            return 0.5  # Neutral
        
        # Sample same timestamps
        common_times = np.intersect1d(player1_data['timestamp'], player2_data['timestamp'])
        
        if len(common_times) < 5:
            return 0.5
        
        p1_subset = player1_data[player1_data['timestamp'].isin(common_times)]
        p2_subset = player2_data[player2_data['timestamp'].isin(common_times)]
        
        # Calculate average distance
        distances = np.sqrt(
            (p1_subset['x'].values - p2_subset['x'].values)**2 +
            (p1_subset['y'].values - p2_subset['y'].values)**2
        )
        
        avg_distance = np.mean(distances)
        
        # Optimal spacing: 4-6 meters
        if 4 <= avg_distance <= 6:
            chemistry = 0.9
        elif 3 <= avg_distance <= 7:
            chemistry = 0.7
        else:
            chemistry = 0.5
        
        return chemistry
    
    def optimize_lineup(self, available_players, positions=None, lineup_size=5):
        """
        Find optimal lineup combination
        
        Args:
            available_players: list of player IDs
            positions: dict of player_id -> position (optional)
            lineup_size: int (default 5)
        
        Returns:
            Best lineup with score
        """
        
        if len(available_players) < lineup_size:
            return {'error': f'Not enough players. Need {lineup_size}, have {len(available_players)}'}
        
        best_lineup = None
        best_score = -999
        
        # Try all combinations
        for lineup in combinations(available_players, lineup_size):
            score = self._evaluate_lineup(lineup)
            
            if score > best_score:
                best_score = score
                best_lineup = lineup
        
        # Get individual ratings
        lineup_details = []
        for player in best_lineup:
            lineup_details.append({
                'player_id': player,
                'rating': self.player_ratings.get(player, 50),
                'position': positions.get(player, 'Unknown') if positions else 'Unknown'
            })
        
        return {
            'lineup': list(best_lineup),
            'score': round(best_score, 2),
            'details': lineup_details,
            'chemistry_score': round(best_score / lineup_size, 2),
            'recommendation': self._generate_lineup_recommendation(lineup_details, best_score)
        }
    
    def _evaluate_lineup(self, lineup):
        """Evaluate lineup quality based on ratings and chemistry"""
        score = 0
        
        # Individual ratings
        for player in lineup:
            score += self.player_ratings.get(player, 50)
        
        # Chemistry bonus
        if self.chemistry_matrix is not None:
            for i, p1 in enumerate(lineup):
                for p2 in lineup[i+1:]:
                    if p1 in self.chemistry_matrix.index and p2 in self.chemistry_matrix.columns:
                        score += self.chemistry_matrix.loc[p1, p2] * 10
        
        return score
    
    def _generate_lineup_recommendation(self, lineup_details, score):
        """Generate recommendation text for lineup"""
        avg_rating = sum(p['rating'] for p in lineup_details) / len(lineup_details)
        
        if avg_rating > 15:
            return "⭐ Lineup ECCELLENTE - Alta efficienza prevista"
        elif avg_rating > 10:
            return "✅ Lineup SOLIDO - Buon bilanciamento"
        elif avg_rating > 5:
            return "⚠️ Lineup ACCETTABILE - Considera alternative"
        else:
            return "🚨 Lineup DEBOLE - Richiede miglioramenti"

# =================================================================
# REAL-TIME GAME ASSISTANT
# =================================================================

class GameAssistant:
    """
    Real-time game assistant using Groq AI
    """
    
    def __init__(self):
        self.game_state = {
            'score_us': 0,
            'score_opponent': 0,
            'quarter': 1,
            'time_remaining': '12:00',
            'current_lineup': [],
            'timeouts_remaining': 7,
            'fouls': {}
        }
    
    def update_game_state(self, **kwargs):
        """Update current game state"""
        self.game_state.update(kwargs)
    
    def ask_assistant(self, question, language='it'):
        """
        Ask the AI assistant a question about game strategy
        
        Args:
            question: str - coach's question
            language: str - response language
        
        Returns:
            AI response with suggestions
        """
        
        # Add context to game state
        context = self.game_state.copy()
        context['score_diff'] = context['score_us'] - context['score_opponent']
        
        # Get Groq response
        response = game_assistant_chat(question, context, language)
        
        return {
            'question': question,
            'answer': response,
            'game_state': self.game_state
        }
    
    def get_timeout_recommendation(self):
        """Determine if timeout is recommended"""
        score_diff = self.game_state['score_us'] - self.game_state['score_opponent']
        
        # Timeout recommended if:
        # - Opponent on run (losing by 8+)
        # - Late game and close (within 5 points, < 2 min)
        # - Need to stop momentum
        
        recommendations = []
        
        if score_diff <= -8:
            recommendations.append("🚨 TIMEOUT CONSIGLIATO - Interrompi run avversario")
        
        if self.game_state['quarter'] >= 4:
            time_parts = self.game_state['time_remaining'].split(':')
            if len(time_parts) == 2:
                minutes = int(time_parts[0])
                if minutes < 2 and abs(score_diff) <= 5:
                    recommendations.append("⏱️ TIMEOUT STRATEGICO - Ultimi 2 minuti, disegna play")
        
        if not recommendations:
            recommendations.append("✅ Situazione sotto controllo - Timeout non urgente")
        
        return recommendations

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def simulate_opponent_stats():
    """Generate realistic opponent stats for testing"""
    return {
        'points_per_game': round(np.random.uniform(95, 115), 1),
        'three_pt_pct': round(np.random.uniform(32, 42), 1),
        'assists_per_game': round(np.random.uniform(20, 28), 1),
        'turnovers_per_game': round(np.random.uniform(12, 18), 1),
        'pace': round(np.random.uniform(95, 105), 1),
        'off_rating': round(np.random.uniform(105, 118), 1),
        'def_rating': round(np.random.uniform(105, 115), 1),
        'rebounds_per_game': round(np.random.uniform(38, 48), 1),
        'ft_pct': round(np.random.uniform(72, 82), 1)
    }
