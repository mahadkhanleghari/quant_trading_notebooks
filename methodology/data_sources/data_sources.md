# Data Sources

## Overview

This section documents the data sources and collection methodologies used across all quantitative research projects.

## Cryptocurrency Data

### Binance API
- **Source**: Binance Public API
- **Asset**: BTC/USDT
- **Frequency**: 1-minute bars
- **Features**: OHLCV + microstructure data
  - Open, High, Low, Close prices
  - Volume and quote volume
  - Trade count per minute
  - Buy/sell ratio (taker buy vs sell volume)
  - Average trade size

### Data Quality
- **Coverage**: 24/7 continuous trading
- **Completeness**: >99.5% data availability
- **Latency**: Real-time with <1 second delay
- **Validation**: Cross-checked with multiple exchanges

## Data Processing Pipeline

1. **Collection**: Automated API calls with rate limiting
2. **Validation**: Outlier detection and missing data handling
3. **Feature Engineering**: Technical indicators and microstructure metrics
4. **Storage**: Efficient CSV format with timestamp indexing

## Simulated Data

For research purposes, realistic market data is generated using:
- **Price Dynamics**: Geometric Brownian Motion with jumps
- **Volume Patterns**: Realistic intraday seasonality
- **Microstructure**: Correlated order flow and spread dynamics
- **Regime Switching**: Multiple volatility and liquidity states

All simulated data maintains statistical properties consistent with real market behavior.
