README

🌀 Monte Carlo Stock Simulator

A simple Python tool to simulate future stock prices using historical data and Monte Carlo methods.

---

💡 What It Does

* Fetches historical price data (Yahoo Finance)
* Calculates daily returns and drawdowns
* Simulates future outcomes using:

  * Bootstrap resampling
  * Normal distribution modeling
* Breaks simulations into **bull**, **bear**, and **neutral** market regimes
* Plots portfolio outcome distributions
* Compares your results to the S\&P 500

---

Example

```bash
$ python monte_carlo_simulator.py
Enter stock ticker symbol (e.g., AAPL): TSLA
Outlook (days): 100
Amount invested: 1000
```

Outputs:

* Chances of beating 10% return or S\&P 500
* Probabilities of gains and drawdowns
* Histograms of simulated future portfolio values

---

📦 Setup

```bash
pip install yfinance pandas numpy matplotlib scipy
```

Then just run:

```bash
python monte_carlo_simulator.py
```

---

⚠️ Disclaimer

This is NOT financial advice. Simulations are based on past performance and assumptions. Use at your own risk.

---


P.S.
This script started as a deep dive into how to best model the probability distribution of a stock’s performance over a given time period (like estimating the chance a stock gains 10% over 50 days). It compares two approaches: one assumes stock returns follow a normal distribution (as in the classic random walk/Brownian motion model), and the other uses bootstrapped historical data. 
