#!/usr/bin/env python3
"""
BTC Data Generator for Microstructure Analysis
Generates realistic BTC/USDT minute data with microstructure features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_btc_data(days=14):
    """Generate realistic BTC/USDT minute data with microstructure features"""
    
    # Time series
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    n_bars = len(timestamps)
    
    # Base price around $95,000
    base_price = 95000
    
    # Generate realistic price series with volatility clustering
    np.random.seed(42)  # Reproducible results
    
    # Price dynamics
    returns = np.random.normal(0, 0.001, n_bars)  # ~0.1% per minute volatility
    
    # Add volatility clustering (GARCH-like)
    vol = np.ones(n_bars) * 0.001
    for i in range(1, n_bars):
        vol[i] = 0.95 * vol[i-1] + 0.05 * abs(returns[i-1]) + 0.0001 * abs(np.random.normal())
        vol[i] = max(vol[i], 0.0001)  # Ensure positive volatility
        returns[i] = np.random.normal(0, vol[i])
    
    # Add intraday patterns
    hours = np.array([ts.hour for ts in timestamps])
    
    # Higher volatility during certain hours (simulate global trading sessions)
    vol_multiplier = 1 + 0.3 * np.sin(2 * np.pi * hours / 24) + 0.1 * np.random.normal(0, 0.1, n_bars)
    returns *= vol_multiplier
    
    # Generate price series
    prices = base_price * np.cumprod(1 + returns)
    
    # OHLC generation
    opens = prices.copy()
    closes = prices * (1 + np.random.normal(0, 0.0002, n_bars))  # Small close variation
    
    # High/Low with realistic spreads
    spreads = np.random.exponential(0.0005, n_bars) + 0.0001  # Realistic crypto spreads
    highs = np.maximum(opens, closes) * (1 + spreads/2)
    lows = np.minimum(opens, closes) * (1 - spreads/2)
    
    # Volume (higher during volatile periods)
    base_volume = 50 + 200 * vol + 100 * np.random.exponential(1, n_bars)
    volume = np.maximum(base_volume, 10)  # Minimum volume
    
    # Trade count (correlated with volume but with noise)
    count = np.maximum(np.round(volume * (0.5 + 0.5 * np.random.random(n_bars))), 1).astype(int)
    
    # Quote volume (price * volume)
    quote_volume = closes * volume
    
    # Trade size (average $ per trade)
    trade_size = quote_volume / count
    
    # Buy/sell ratio (mean-reverting around 0.5 with trends)
    buy_sell_ratio = np.zeros(n_bars)
    buy_sell_ratio[0] = 0.5
    
    for i in range(1, n_bars):
        # Mean-reverting with momentum
        momentum = 0.1 * returns[i] / vol[i] if vol[i] > 0 else 0
        buy_sell_ratio[i] = (0.9 * buy_sell_ratio[i-1] + 
                            0.1 * 0.5 + 
                            momentum + 
                            0.05 * np.random.normal())
        buy_sell_ratio[i] = np.clip(buy_sell_ratio[i], 0.1, 0.9)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume,
        'count': count,
        'trade_size': trade_size,
        'buy_sell_ratio': buy_sell_ratio,
        'quote_volume': quote_volume
    })
    
    return df

def main():
    """Generate and save BTC data"""
    
    # Create data directory
    data_dir = '../../data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data
    print("Generating BTC/USDT sample data...")
    df = generate_realistic_btc_data(days=14)
    
    # Save to CSV
    output_file = os.path.join(data_dir, 'BTC_minute_data.csv')
    df.to_csv(output_file, index=False)
    
    # Summary
    print(f"Data generated: {len(df):,} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"File saved: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    print(f"\\nMarket features:")
    print(f"- Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    print(f"- Average trades per minute: {df['count'].mean():.1f}")
    print(f"- Average trade size: ${df['trade_size'].mean():.2f}")
    print(f"- Buy/sell ratio range: {df['buy_sell_ratio'].min():.3f} - {df['buy_sell_ratio'].max():.3f}")
    print(f"- Daily volatility: {df['close'].pct_change().std() * np.sqrt(1440):.1%}")
    
    print(f"\\nReady for microstructure analysis!")

if __name__ == "__main__":
    main()
