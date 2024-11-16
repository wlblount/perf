#perf.py    v1.0.1   11/16/2004   11:25PM

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fmp import fmp_price

def metrics(prices_df, start=None, end=None):

    '''
    Calculate performance metrics for a given time series of prices and visualize the results.

    This function computes various performance metrics for a time series of asset prices, 
    including cumulative returns, Sharpe Ratio, Sortino Ratio, and more. It also generates 
    plots to visualize cumulative returns, drawdowns, and daily returns over the specified 
    date range.

    Parameters
    ----------
    prices_df : pd.DataFrame
        A DataFrame with a datetime index and a single numeric column representing asset prices.
    start : str, optional
        The start date for the analysis period in 'YYYY-MM-DD' format (default is '2023-11-14').
    end : str, optional
        The end date for the analysis period in 'YYYY-MM-DD' format (default is '2024-11-14').

    Returns
    -------
    None
        Prints calculated performance metrics and displays plots for cumulative returns, 
        drawdown series, and daily returns.

    Raises
    ------
    ValueError
        If the DataFrame does not have a datetime index or a numeric column, or if no risk-free 
        rate data is available for the specified date range.

    Metrics Calculated
    ------------------
    - Risk Free Rate (Ann.): The average annualized risk-free rate (3-month T-Bill).
    - Risk Free Rate (Adj.): Adjusted risk-free rate based on the number of trading days.
    - Cumulative Return: The total return over the specified period.
    - Excess Return: The return above the risk-free rate.
    - Sharpe Ratio: Excess return divided by annual volatility.
    - Sortino Ratio: Excess return divided by the downside deviation.
    - Gain-to-Pain Ratio: Ratio of positive returns to absolute negative returns.
    - Return over Max Drawdown (ret2dd): Return divided by the maximum drawdown.
    - Max Drawdown: The maximum observed loss from a peak to a trough.
    - Annual Volatility: Standard deviation of returns annualized.

    Plots Generated
    ---------------
    1. Cumulative Returns: Line plot of cumulative returns over the specified date range.
    2. Drawdown Series: Line plot of drawdowns with shaded regions indicating drawdown periods.
    3. Daily Returns: Bar plot of daily returns with green bars for positive returns and red bars for negative returns.

    Example
    -------
    >>> import pandas as pd
    >>> from fmp import fmp_price
    >>> df = pd.DataFrame({'bnchmk': [100, 101, 102, 103, 102]}, 
                          index=pd.date_range(start='2023-11-10', periods=5, freq='D'))
    >>> metrics(df, start='2023-11-10', end='2023-11-14')
    '''

    
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a datetime index.")
    
    numeric_columns = prices_df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        raise ValueError("The DataFrame must contain at least one numeric column.")
    
    # Fetch the risk-free rate (3-month T-Bill rate)
    riskFree_raw = fmp_price('^IRX', start=start, end=end)
    print('Number of observations in risk-free rate data:', len(riskFree_raw))
    
    if len(riskFree_raw) == 0:
        raise ValueError("No risk-free rate data available for the specified date range.")
    
    riskFree_mean = riskFree_raw.mean()[0] / 100  # Convert to decimal form
    risk_free_days = len(riskFree_raw)
    
    # Adjust the risk-free rate for partial years
    if risk_free_days < 252:
        riskFree = riskFree_mean * (risk_free_days / 252)
    else:
        riskFree = riskFree_mean

    # Ensure the DataFrame index is timezone-aware (convert or localize to UTC)
    if prices_df.index.tz is None:
        prices_df.index = prices_df.index.tz_localize('UTC')
    else:
        prices_df.index = prices_df.index.tz_convert('UTC')

    # Convert start and end to timezone-aware datetime with UTC
    start = pd.to_datetime(start).tz_localize('UTC')
    if end is not None:
        end = pd.to_datetime(end).tz_localize('UTC')
    else:
        end = prices_df.index.max()
    
    # Check if the start and end dates are within the DataFrame's date range
    if start < prices_df.index.min() or end > prices_df.index.max():
        raise ValueError(f"Date range out of bounds: {prices_df.index.min().date()} to {prices_df.index.max().date()}.")

    prices_df = prices_df.loc[start:end]
    returns = prices_df[numeric_columns[0]].pct_change().dropna()
    
    if returns.empty:
        print("No returns available for the specified date range.")
        return

    cumulative_returns = (1 + returns).cumprod() - 1
    excessReturns = cumulative_returns.iloc[-1] - riskFree
    annual_volatility = returns.std() * np.sqrt(252)
    
    sharpe_ratio = excessReturns / annual_volatility
    
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (excessReturns / downside_deviation) if downside_deviation != 0 else None
    
    sum_positive_returns = returns[returns > 0].sum()
    sum_negative_returns = abs(returns[returns < 0].sum())
    gain_to_pain_ratio = (sum_positive_returns / sum_negative_returns) if sum_negative_returns != 0 else None
    
    expanding_max = cumulative_returns.cummax()
    drawdown_series = cumulative_returns - expanding_max
    max_drawdown = drawdown_series.min()
    
    ret2dd = (cumulative_returns / abs(max_drawdown)) if max_drawdown != 0 else None

    metrics = {
        "Risk Free Rate (Ann.)": riskFree_mean,
        "Risk Free Rate (Adj.)": riskFree,
        "Cum. Return": cumulative_returns.iloc[-1],
        "Excess Return (End Value)": excessReturns,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Gain-to-Pain Ratio": gain_to_pain_ratio,
        "Return over Max Drawdown (ret2dd)": ret2dd.iloc[-1] if isinstance(ret2dd, pd.Series) else ret2dd,
        "Max Drawdown": max_drawdown,
        "Annual Volatility": annual_volatility
    }
    
    print(f"\nPerformance Metrics ({start.date()} to {end.date()}):\n" + "-" * 50)
    for key, value in metrics.items():
        print(f"{key:45}: {value:.4f}" if isinstance(value, (int, float)) else f"{key:45}: {value}")
    print("\n"+"")
    print("\n"+"")

    # Plot cumulative returns, drawdown, and daily returns
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax[0].plot(cumulative_returns, label='Cumulative Returns', color='blue')
    ax[0].set_title(f'Cumulative Returns ({start.date()} to {end.date()})')
    ax[0].grid(True)
    ax[0].legend()

        # Plot drawdown series with shading
    ax[1].plot(drawdown_series, label='Drawdown', color='red')
    ax[1].fill_between(drawdown_series.index, drawdown_series, 0, where=(drawdown_series < 0), color='red', alpha=0.3)
    ax[1].set_title(f'Drawdown Series ({start.date()} to {end.date()})')
    ax[1].grid(True)
    ax[1].legend()

    colors = ['green' if r > 0 else 'red' for r in returns]
    ax[2].bar(returns.index, returns, color=colors, width=0.8)
    ax[2].set_title(f'Daily Returns ({start.date()} to {end.date()})')
    ax[2].grid(True)

    ax[2].xaxis.set_major_locator(mdates.MonthLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
