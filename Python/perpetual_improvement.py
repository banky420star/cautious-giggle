import json
import os
import time
from datetime import datetime
from collections import defaultdict, deque
from loguru import logger
import numpy as np


class PerpetualImprovementSystem:
    """
    Continuously adjusts learning rates for LSTM, PPO, and DreamerV3 models based on performance.
    Uses pattern recognition success to inform learning parameter adjustments.
    Maintains adaptation history to track improvement over time.
    """
    
    def __init__(self, adaptation_history_size=100):
        self.adaptation_history_size = adaptation_history_size
        self.adaptation_history = defaultdict(lambda: deque(maxlen=adaptation_history_size))
        self.last_improvement_action = {}
        self.model_performance = defaultdict(list)
        self.pattern_success_rates = defaultdict(lambda: defaultdict(list))
        self.learning_rates = defaultdict(dict)
        
        # Initialize with default learning rates
        self._initialize_default_learning_rates()
        
    def _initialize_default_learning_rates(self):
        """Initialize default learning rates for each model type"""
        self.learning_rates['lstm'] = {'base_lr': 1e-3}
        self.learning_rates['ppo'] = {'base_lr': 1e-4}
        self.learning_rates['dreamer'] = {
            'world_model_lr': 3e-4,
            'actor_lr': 1e-4,
            'critic_lr': 3e-4
        }
        
    def record_pattern_success(self, pattern_name, market_regime, success_rate):
        """
        Record the success rate of a pattern in a specific market regime
        :param pattern_name: Name of the pattern detected
        :param market_regime: Market regime (high_vol_bull, low_vol_bear, etc.)
        :param success_rate: Success rate of the pattern (0.0 to 1.0)
        """
        self.pattern_success_rates[pattern_name][market_regime].append({
            'timestamp': time.time(),
            'success_rate': success_rate
        })
        
        # Keep only recent history
        if len(self.pattern_success_rates[pattern_name][market_regime]) > 50:
            self.pattern_success_rates[pattern_name][market_regime] = self.pattern_success_rates[pattern_name][market_regime][-50:]
                
        logger.debug(f"Recorded pattern success: {pattern_name} in {market_regime}: {success_rate}")
        
    def get_pattern_success_rate(self, pattern_name, market_regime, window_size=10):
        """
        Get the average success rate of a pattern in a specific market regime
        :param pattern_name: Name of the pattern
        :param market_regime: Market regime
        :param window_size: Number of recent records to consider
        :return: Average success rate
        """
        if pattern_name not in self.pattern_success_rates or \
           market_regime not in self.pattern_success_rates[pattern_name]:
            return 0.5  # Default neutral success rate
            
        records = self.pattern_success_rates[pattern_name][market_regime]
        if not records:
            return 0.5
            
        recent_records = records[-window_size:] if len(records) >= window_size else records
        avg_success = np.mean([r['success_rate'] for r in recent_records])
        return avg_success
        
    def record_model_performance(self, model_type, symbol, performance_metric):
        """
        Record performance metric for a model
        :param model_type: Type of model (lstm, ppo, dreamer)
        :param symbol: Trading symbol
        :param performance_metric: Performance metric (e.g., accuracy, profit factor)
        """
        key = f"{model_type}_{symbol}"
        self.model_performance[key].append({
            'timestamp': time.time(),
            'metric': performance_metric
        })
        
        # Keep only recent history
        if len(self.model_performance[key]) > 100:
            self.model_performance[key] = self.model_performance[key][-100:]
            
        logger.debug(f"Recorded {model_type} performance for {symbol}: {performance_metric}")
        
    def get_model_performance_trend(self, model_type, symbol, window_size=10):
        """
        Get the performance trend for a model
        :param model_type: Type of model
        :param symbol: Trading symbol
        :param window_size: Number of recent records to consider
        :return: Trend (-1.0 to 1.0, where negative means declining performance)
        """
        key = f"{model_type}_{symbol}"
        if key not in self.model_performance or not self.model_performance[key]:
            return 0.0  # No trend data
            
        records = self.model_performance[key]
        if len(records) < 2:
            return 0.0  # Not enough data for trend
            
        recent_records = records[-window_size:] if len(records) >= window_size else records
        if len(recent_records) < 2:
            return 0.0
            
        # Calculate linear trend
        metrics = [r['metric'] for r in recent_records]
        x_vals = list(range(len(metrics)))
        
        if len(metrics) >= 2:
            # Simple linear regression slope
            n = len(metrics)
            sum_x = sum(x_vals)
            sum_y = sum(metrics)
            sum_xy = sum(x * y for x, y in zip(x_vals, metrics))
            sum_x2 = sum(x * x for x in x_vals)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                # Normalize slope to -1.0 to 1.0 range
                max_metric = max(metrics) if metrics else 1.0
                min_metric = min(metrics) if metrics else 0.0
                range_metric = max_metric - min_metric if max_metric != min_metric else 1.0
                normalized_slope = slope / (range_metric / len(metrics)) if range_metric != 0 else 0.0
                # Clamp to [-1, 1]
                trend = max(-1.0, min(1.0, normalized_slope))
                return trend
                
        return 0.0
        
    def adjust_learning_rate(self, model_type, symbol=None):
        """
        Adjust learning rate based on performance trends and pattern success
        :param model_type: Type of model (lstm, ppo, dreamer)
        :param symbol: Trading symbol (optional)
        :return: Dictionary of adjusted learning rates
        """
        # Get performance trend
        trend = self.get_model_performance_trend(model_type, symbol) if symbol else 0.0
        
        # Get pattern-based adjustment
        pattern_adjustment = self._get_pattern_based_adjustment(model_type, symbol)
        
        # Combine adjustments
        total_adjustment = (trend * 0.7) + (pattern_adjustment * 0.3)
        
        # Apply adjustment to learning rates
        adjusted_rates = {}
        base_rates = self.learning_rates.get(model_type, {})
        
        for param_name, base_lr in base_rates.items():
            # Adjust learning rate based on performance
            # If performance is improving (positive trend), we can increase LR slightly
            # If performance is declining (negative trend), we should decrease LR
            adjustment_factor = 1.0 + (total_adjustment * 0.2)  # Max 20% adjustment
            adjustment_factor = max(0.5, min(2.0, adjustment_factor))  # Clamp to 0.5x - 2.0x
            
            adjusted_lr = base_lr * adjustment_factor
            adjusted_rates[param_name] = adjusted_lr
            
        # Store the adjustment action
        action_key = f"{model_type}_{symbol}" if symbol else model_type
        self.last_improvement_action[action_key] = {
            'timestamp': time.time(),
            'model_type': model_type,
            'symbol': symbol,
            'performance_trend': trend,
            'pattern_adjustment': pattern_adjustment,
            'total_adjustment': total_adjustment,
            'adjusted_learning_rates': adjusted_rates.copy(),
            'base_learning_rates': base_rates.copy()
        }
        
        # Record in adaptation history
        self.adaptation_history[action_key].append({
            'timestamp': time.time(),
            'action': self.last_improvement_action[action_key].copy()
        })
        
        logger.info(f"Adjusted {model_type} learning rate{symbol if symbol else ''}: "
                   f"trend={trend:.3f}, pattern_adj={pattern_adjustment:.3f}, "
                   f"total_adj={total_adjustment:.3f}")
                   
        return adjusted_rates
        
    def _get_pattern_based_adjustment(self, model_type, symbol):
        """
        Calculate learning rate adjustment based on pattern success rates
        :param model_type: Type of model
        :param symbol: Trading symbol
        :return: Adjustment factor (-1.0 to 1.0)
        """
        # Placeholder: returns neutral adjustment for now
        return 0.0
        
    def get_adaptation_history(self, model_type=None, symbol=None):
        """
        Get adaptation history for a model/symbol combination
        :param model_type: Type of model (optional)
        :param symbol: Trading symbol (optional)
        :return: List of adaptation records
        """
        if model_type and symbol:
            key = f"{model_type}_{symbol}"
            return list(self.adaptation_history.get(key, []))
        elif model_type:
            # Return history for all symbols of this model type
            history = []
            for key, records in self.adaptation_history.items():
                if key.startswith(f"{model_type}_"):
                    history.extend(records)
            return history
        else:
            # Return all history
            history = []
            for records in self.adaptation_history.values():
                history.extend(records)
            return history
            
    def get_last_improvement_action(self, model_type=None, symbol=None):
        """
        Get the last improvement action for a model/symbol combination
        :param model_type: Type of model (optional)
        :param symbol: Trading symbol (optional)
        :return: Last improvement action record
        """
        if model_type and symbol:
            key = f"{model_type}_{symbol}"
            return self.last_improvement_action.get(key, {})
        elif model_type:
            # Return most recent action for this model type
            actions = []
            for key, action in self.last_improvement_action.items():
                if key.startswith(f"{model_type}_"):
                    actions.append((key, action))
            if actions:
                # Sort by timestamp and return most recent
                actions.sort(key=lambda x: x[1].get('timestamp', 0), reverse=True)
                return actions[0][1]
            return {}
        else:
            # Return most recent action overall
            if not self.last_improvement_action:
                return {}
            most_recent_key = max(self.last_improvement_action.keys(), 
                                key=lambda k: self.last_improvement_action[k].get('timestamp', 0))
            return self.last_improvement_action[most_recent_key]
            
    def save_state(self, filepath):
        """
        Save the perpetual improvement system state to a file
        :param filepath: Path to save the state
        """
        state = {
            'adaptation_history': {
                k: list(v) for k, v in self.adaptation_history.items()
            },
            'last_improvement_action': self.last_improvement_action,
            'model_performance': {
                k: v for k, v in self.model_performance.items()
            },
            'pattern_success_rates': {
                k: {kk: vv for kk, vv in v.items()} 
                for k, v in self.pattern_success_rates.items()
            },
            'learning_rates': dict(self.learning_rates)
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved perpetual improvement system state to {filepath}")
        
    def load_state(self, filepath):
        """
        Load the perpetual improvement system state from a file
        :param filepath: Path to load the state from
        """
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} does not exist, initializing with defaults")
            return
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.adaptation_history = defaultdict(lambda: deque(maxlen=self.adaptation_history_size))
            for k, v in state.get('adaptation_history', {}).items():
                self.adaptation_history[k] = deque(v, maxlen=self.adaptation_history_size)
            self.last_improvement_action = state.get('last_improvement_action', {})
            self.model_performance = defaultdict(list, state.get('model_performance', {}))
            self.pattern_success_rates = defaultdict(lambda: defaultdict(list), {
                k: defaultdict(list, vv) for k, vv in state.get('pattern_success_rates', {}).items()
            })
            self.learning_rates = defaultdict(dict, state.get('learning_rates', {}))
            logger.info(f"Loaded perpetual improvement system state from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load perpetual improvement state: {e}")
            self._initialize_default_learning_rates()


# Global singleton
_pis_singleton = PerpetualImprovementSystem()

def get_perpetual_improvement_system():
    return _pis_singleton

def export_perpetual_improvement_state() -> dict:
    """Export a lightweight snapshot of the perpetual improvement state for UI/monitoring."""
    pis = get_perpetual_improvement_system()
    try:
        return {
            'last_improvement_action': pis.get_last_improvement_action(),
            'adaptation_history': pis.get_adaptation_history(),
            'learning_rates': dict(pis.learning_rates)
        }
    except Exception:
        return {
            'last_improvement_action': {},
            'adaptation_history': [],
            'learning_rates': {}
        }
