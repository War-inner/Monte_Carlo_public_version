import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def fetch_stock_data(ticker, years=4, end_date="2025-05-19"):
    """
    Fetch historical stock data for a given ticker and time frame.
    """
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.DateOffset(years=years)
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return hist


def compute_max_drawdown(return_path):
    """
    Compute the maximum drawdown from a series of multiplicative returns.
    """
    cumulative = np.cumprod(return_path)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    return drawdowns.min()


def compute_daily_returns(prices):
    """
    Calculate daily simple returns from a price series.
    """
    return prices.pct_change().dropna()


def monte_carlo_bootstrap_simulation(returns, outlook=100, invested=100, iterations=100000, sp500_benchmark=None):
    """
    Perform a bootstrap Monte Carlo simulation by resampling historical returns.
    """
    pct_returns = returns + 1
    outcomes = []
    drawdowns = []

    for _ in range(iterations):
        simulated_path = np.random.choice(pct_returns, size=outlook, replace=True)
        final_value = np.prod(simulated_path) * invested
        max_dd = compute_max_drawdown(simulated_path)
        outcomes.append(final_value)
        drawdowns.append(max_dd)

    return summarize_simulation_results(
        method_name='Bootstrap',
        outcomes=outcomes,
        outlook=outlook,
        invested=invested,
        iterations=iterations,
        drawdowns=drawdowns,
        benchmark=sp500_benchmark
    )


def monte_carlo_parametric_simulation(returns, outlook, invested=100, iterations=10000,
                                     distribution='Normal', sp500_benchmark=None):
    """
    Perform a parametric Monte Carlo simulation fitting returns to a specified distribution.
    Supported distributions: Normal, Gamma, Beta.
    """
    pct_returns = returns + 1

    if distribution == 'Normal':
        params = stats.norm.fit(pct_returns)
        sampler = lambda size: np.random.normal(*params, size)
        dist_name = 'norm'
        method_name = 'Normal'
    elif distribution == 'Gamma':
        params = stats.gamma.fit(pct_returns, floc=0)
        sampler = lambda size: np.random.gamma(params[0], params[2], size)
        dist_name = 'gamma'
        method_name = 'Gamma'
    elif distribution == 'Beta':
        params = stats.beta.fit(pct_returns)
        sampler = lambda size: stats.beta.rvs(*params, size=size)
        dist_name = 'beta'
        method_name = 'Beta'
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    _, p_value = stats.kstest(pct_returns, dist_name, args=params)

    outcomes = []
    drawdowns = []

    for _ in range(iterations):
        simulated_path = sampler(outlook)
        final_value = np.prod(simulated_path) * invested
        max_dd = compute_max_drawdown(simulated_path)
        outcomes.append(final_value)
        drawdowns.append(max_dd)

    return summarize_simulation_results(
        method_name=method_name,
        outcomes=outcomes,
        outlook=outlook,
        invested=invested,
        iterations=iterations,
        drawdowns=drawdowns,
        benchmark=sp500_benchmark,
        p_value=p_value
    )


def summarize_simulation_results(method_name, outcomes, outlook, invested, iterations,
                                 drawdowns=None, benchmark=None, p_value=0):
    """
    Summarize Monte Carlo simulation results, including probabilities and statistics.
    """
    chance_positive = 1 - compute_percentile(outcomes, invested)
    threshold_10pct = invested * (1 + (0.10 / 252) * outlook)  # Approximate 10% annualized
    chance_beat_10pct = 1 - compute_percentile(outcomes, threshold_10pct)

    chance_beat_benchmark = None
    if benchmark is not None:
        chance_beat_benchmark = 1 - compute_percentile(outcomes, benchmark)

    prob_drawdown_exceed = None
    if drawdowns is not None:
        dd_threshold = -0.10  # 10% max drawdown threshold
        prob_drawdown_exceed = np.mean([dd <= dd_threshold for dd in drawdowns])

    return {
        'name': method_name,
        'p-value': p_value,
        'Chance to Beat 10%': chance_beat_10pct,
        'Chance to Beat SP500': chance_beat_benchmark,
        'Chance of Positive Return': chance_positive,
        'Probability Drawdown > 10%': prob_drawdown_exceed,
        'Outcomes': outcomes
    }


def plot_histogram(data, title):
    """
    Plot a histogram of the portfolio values with a mean line and title.
    """
    plt.hist(data, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(data), color='green', linestyle='-', label='Mean estimate')
    plt.title(title)
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def compute_percentile(data, value):
    """
    Calculate the percentile rank of a value within a data array.
    """
    sorted_data = np.sort(data)
    idx = np.searchsorted(sorted_data, value, side='right')
    return idx / len(sorted_data)


def create_summary_dataframe(results_df):
    """
    Create a DataFrame summarizing simulation results for each method.
    """
    summary_df = pd.DataFrame()
    for _, row in results_df.iterrows():
        method = row['name']
        outcomes = row['Outcomes']
        desc = pd.Series(outcomes).describe()
        summary_df[method] = desc
    return summary_df


def identify_market_regimes(returns, window=40, threshold=0.00015):
    """
    Classify returns into market regimes: 'bull', 'bear', or 'neutral' based on rolling average.
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    rolling_avg = returns.rolling(window=window).mean()
    regimes = rolling_avg.apply(lambda x: 'bull' if x > threshold else ('bear' if x < -threshold else 'neutral'))

    return pd.DataFrame({'Returns': returns, 'Regime': regimes})


def run_simulations_by_market_regime(returns_vector, outlook, invested_amount, market='None', sp500_benchmark=None):
    """Run Monte Carlo simulations segmented by market regime and display results.
    """
    regime_df = identify_market_regimes(returns_vector)

    if market == 'bull':
        filtered_returns = regime_df[regime_df['Regime'] == 'bull']['Returns']
    elif market == 'bear':
        filtered_returns = regime_df[regime_df['Regime'] == 'bear']['Returns']
    elif market == 'neutral':
        filtered_returns = regime_df[regime_df['Regime'] == 'neutral']['Returns']
    else:
        filtered_returns = returns_vector

    print(f'\n---- {market if market != "None" else "Full"} Market ----')

    results = []

    # Bootstrap simulation
    results.append(
        monte_carlo_bootstrap_simulation(filtered_returns, invested=invested_amount, outlook=outlook,
                                         sp500_benchmark=sp500_benchmark)
    )

    # Parametric simulations for all supported distributions
    for dist in ['Normal', 'Gamma', 'Beta']:
        results.append(
            monte_carlo_parametric_simulation(filtered_returns, invested=invested_amount, outlook=outlook,
                                             distribution=dist, sp500_benchmark=sp500_benchmark)
        )

    results_df = pd.DataFrame(results)

    for _, row in results_df.iterrows():
        plot_histogram(row['Outcomes'], f"{row['name']} Simulation Portfolio Values")

    summary_df = create_summary_dataframe(results_df)
    print(results_df.drop(columns=['Outcomes']))
    print(summary_df)

    return results_df, summary_df


def main():
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()

    try:
        stock_data = fetch_stock_data(ticker)
        if stock_data.empty:
            print("No data found for the ticker.")
            return

        outlook = int(input('Outlook (days): '))
        invested_amount = int(input('Amount invested: '))

        sp500_data = fetch_stock_data('SPY')
        sp500_returns = compute_daily_returns(sp500_data['Close']).values
        sp500_cum_return = np.prod(sp500_returns[-outlook:] + 1)
        sp500_benchmark_value = sp500_cum_return * invested_amount

        close_prices = stock_data['Close']
        daily_returns = compute_daily_returns(close_prices)
        returns_vector = daily_returns.values

        #Run simulations for each market regime plus full market
        run_simulations_by_market_regime(returns_vector, outlook, invested_amount, 'bull', sp500_benchmark_value)
        run_simulations_by_market_regime(returns_vector, outlook, invested_amount, 'bear', sp500_benchmark_value)
        run_simulations_by_market_regime(returns_vector, outlook, invested_amount, 'neutral', sp500_benchmark_value)
        run_simulations_by_market_regime(returns_vector, outlook, invested_amount, 'None', sp500_benchmark_value)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
