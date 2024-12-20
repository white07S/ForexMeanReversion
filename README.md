Project Overview

ForexMeanReversion is a Python-based quantitative trading framework designed to exploit mean-reversion dynamics in the EUR/USD currency pair. The strategy is founded on a stochastic model—specifically, the Ornstein-Uhlenbeck (OU) process—to model the log-price behavior, and it integrates a machine learning layer to capture time-varying parameters. The combined approach allows for robust adaptation to changing market regimes and the continuous exploitation of mean-reverting tendencies in FX prices.

This project aims to provide:
	•	A rigorous mathematical foundation using stochastic calculus (Itô calculus) and maximum likelihood estimation (MLE) for parameter inference.
	•	Adaptive parameter modeling via a neural network to handle nonstationarities in real-world financial time series.
	•	A fully integrated pipeline: from raw data preparation, parameter estimation, machine learning–driven parameter updates, signal generation, to backtesting and performance analysis.

Project Structure

The repository is organized as follows:

.
├── ForexMeanReversion
│   ├── __init__.py
│   ├── backtest
│   │   ├── __init__.py
│   │   ├── backtester.py
│   │   ├── performance.py
│   │   └── signal_generation.py
│   ├── config.py
│   ├── data
│   │   ├── __init__.py
│   │   └── data_preparation.py
│   ├── main.py
│   └── ou_model
│       ├── __init__.py
│       ├── features.py
│       ├── ml_model.py
│       ├── ou_estimation.py
│       └── ou_params.py
├── LICENSE
├── README.md
└── setup.py

Key Directories:
	•	ForexMeanReversion/data: Data ingestion and cleaning.
	•	ForexMeanReversion/ou_model: Core mathematical modeling (Ornstein-Uhlenbeck), parameter estimation, feature engineering, and ML model definitions.
	•	ForexMeanReversion/backtest: Signal generation, strategy simulation, and performance evaluation.
	•	ForexMeanReversion: Main configuration file and a main.py entry point that runs the entire pipeline.

Mathematical Foundations

1. Mean-Reversion and the Ornstein-Uhlenbeck (OU) Process

The fundamental assumption of this strategy is that the log-price of EUR/USD, denoted ￼, exhibits mean-reverting behavior. A continuous-time Ornstein-Uhlenbeck (OU) process is often used to model such dynamics. The OU process is defined by the following Stochastic Differential Equation (SDE):

￼

Key parameters:
	•	￼: The long-term mean (equilibrium) to which the process tends to revert.
	•	￼: The mean-reversion speed; how strongly the process is pulled back towards ￼.
	•	￼: The volatility parameter controlling the diffusion around the mean.

Properties of the OU Process:
	1.	Mean-Reversion: The drift term ￼ ensures that if ￼ is above ￼, the process tends to decrease, and if ￼ is below ￼, it tends to increase. The parameter ￼ controls how fast this convergence happens.
	2.	Stationary Distribution: The OU process has a stationary distribution which is Normal with:
￼
at long times. This stationary distribution underpins the idea of mean reversion, as the process does not drift off to infinity but hovers around ￼.

2. Discretization and Maximum Likelihood Estimation (MLE)

In practice, we observe discrete time series (e.g., 1-minute bars). We must discretize the continuous-time OU process. For a small time step ￼, the OU process can be discretized as:

￼

where ￼.

Mean and Variance of the Discrete OU Increment:

Given ￼, the conditional distribution of ￼ is normal with:
￼
￼

Maximum Likelihood Estimation:

Given a time series ￼ and the parametric form of the OU process, we estimate ￼ by maximizing the log-likelihood:

￼

where ￼ is the normal density with the mean and variance given above.

The negative log-likelihood to be minimized is:

￼

We use numerical optimization (quasi-Newton methods) for MLE.

3. Time-Varying Parameters and Machine Learning

Financial markets are nonstationary. The OU parameters ￼ might evolve over time due to regime changes, shifts in monetary policy, or evolving market microstructure.

Approach:
	1.	Rolling Window Estimation: We repeatedly estimate ￼ on rolling windows of historical data. This gives us time series of estimated parameters ￼.
	2.	Feature Engineering: Construct a feature vector ￼ from recent price data. For example:
	•	Short-term moving average of log-prices ￼.
	•	Long-term moving average ￼.
	•	Short-term volatility of returns ￼.
	•	Long-term volatility of returns ￼.
Thus:
￼
	3.	Neural Network Regression:
We train a feed-forward neural network ￼ to predict ￼ from ￼:
￼
The loss function is the Mean Squared Error (MSE):
￼
By minimizing ￼, the NN learns how features relate to changing parameters, capturing nonstationarities in real time.

4. Trading Strategy and Signal Generation

Once we have time-varying parameters ￼, we use the OU model’s equilibrium level (￼) and the stationary standard deviation ￼ to generate signals.

Signal Logic:
	1.	Compute the deviation:
￼
	2.	Compare ￼ to ￼.
	3.	If ￼, go short (price is “too high” and should revert down).
	4.	If ￼, go long (price is “too low” and should revert up).
	5.	Otherwise, stay flat.

Choosing ￼: This threshold controls sensitivity. A typical value might be ￼ or ￼.

5. Backtesting and Performance Metrics

We simulate trades on historical data using predicted parameters and signals. The P&L (profit and loss) at each step is derived from the position and subsequent price changes. We incorporate transaction costs to ensure realism.

Performance Metrics:
	•	Sharpe Ratio:
￼
where ￼ are the strategy returns per bar.
	•	Max Drawdown:
\[
\text{MDD} = \max_{t}\bigl(\frac{\text{Peak up to time } t - \text{Equity}_t}{\text{Peak up to time } t}\bigr)
\]
	•	Hit Ratio: Probability of profitable trades.

These metrics help assess the stability and attractiveness of the strategy.

Step-by-Step Workflow
	1.	Data Preparation (data_preparation.py):
	•	Merges raw CSV files of EUR/USD 1-min quotes.
	•	Cleans and transforms prices into log-prices.
	2.	Baseline Parameter Estimation (ou_estimation.py):
	•	Uses MLE on a large historical window to get a baseline ￼.
	3.	Rolling Window Estimation:
	•	Computes parameters ￼ on rolling windows to create training labels for the ML model.
	4.	Feature Engineering (features.py):
	•	Constructs feature vectors ￼ using moving averages and volatilities.
	5.	ML Model Training (ml_model.py):
	•	Trains a neural network to map ￼.
	6.	Online Parameter Prediction:
	•	Uses the trained NN model to predict parameters as time moves forward.
	7.	Signal Generation (signal_generation.py):
	•	Generates trading signals based on the deviation from the mean.
	8.	Backtesting (backtester.py):
	•	Simulates the strategy over test data.
	•	Computes P&L and performance metrics (performance.py).
	9.	Analysis:
	•	Review Sharpe ratio, drawdowns, hit ratios.
	•	Evaluate if strategy improves over a naive mean-reversion approach.

