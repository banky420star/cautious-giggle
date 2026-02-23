import random
import pandas as pd
import numpy as np
import pandas_ta as ta

class StrategyEvolver:
    """Genetic Algorithm to discover optimal indicator combinations."""
    def __init__(self, df):
        self.df = df
        self.indicators = ['RSI', 'MACD', 'EMA', 'BBANDS', 'ATR', 'ADX', 'STOCH']
        self.population_size = 20
        self.generations = 5

    def generate_random_strategy(self):
        """A strategy is a set of indicator parameters."""
        return {
            'rsi_period': random.randint(7, 21),
            'ema_fast': random.randint(5, 20),
            'ema_slow': random.randint(21, 50),
            'bb_std': random.uniform(1.5, 2.5),
            'macd_fast': random.randint(8, 16),
            'macd_slow': random.randint(20, 32),
            'macd_signal': random.randint(7, 12)
        }

    def evaluate_fitness(self, strategy):
        """Backtest the strategy parameters and return a 'fitness' score (Sharpe Ratio)."""
        df = self.df.copy()
        
        # Apply indicators with strategy params
        df.ta.rsi(length=strategy['rsi_period'], append=True)
        df.ta.ema(length=strategy['ema_fast'], append=True)
        df.ta.ema(length=strategy['ema_slow'], append=True)
        df.ta.macd(fast=strategy['macd_fast'], slow=strategy['macd_slow'], signal=strategy['macd_signal'], append=True)
        
        # Simple crossover logic for fitness testing
        ema_fast_col = f"EMA_{strategy['ema_fast']}"
        ema_slow_col = f"EMA_{strategy['ema_slow']}"
        
        if ema_fast_col not in df.columns or ema_slow_col not in df.columns:
            return -1
            
        df['signal'] = np.where(df[ema_fast_col] > df[ema_slow_col], 1, -1)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() != 0 else -1
        return sharpe if not np.isnan(sharpe) else -1

    def evolve(self):
        """Run the GA evolution."""
        population = [self.generate_random_strategy() for _ in range(self.population_size)]
        
        for gen in range(self.generations):
            fitness_scores = [(self.evaluate_fitness(s), s) for s in population]
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            print(f"Gen {gen}: Best Sharpe = {fitness_scores[0][0]:.4f}")
            
            # Selection (Top 5)
            parents = [s for _, s in fitness_scores[:5]]
            
            # Crossover & Mutation
            new_population = parents.copy()
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = {k: random.choice([p1[k], p2[k]]) for k in p1.keys()}
                
                # Mutation
                if random.random() < 0.1:
                    key = random.choice(list(child.keys()))
                    if isinstance(child[key], int):
                        child[key] += random.choice([-1, 1])
                    else:
                        child[key] += random.uniform(-0.1, 0.1)
                
                new_population.append(child)
            
            population = new_population
            
        return population[0] # Return best strategy
