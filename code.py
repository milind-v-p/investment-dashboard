import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path, index_col=None)
    return data

# Extract mean return and volatility from dataset
def extract_metrics(data, tickers):
    metrics = {}
    for ticker in tickers:
        try:
            close_col = f"{ticker}.3"  # Assume close price is in column ticker.3
            close_prices = pd.to_numeric(data[close_col], errors='coerce').dropna()
            daily_returns = close_prices.pct_change().dropna()
            mean_return = daily_returns.mean()
            volatility = daily_returns.std()
            metrics[ticker] = {
                "mean_return": mean_return,
                "volatility": volatility,
                "daily_returns": daily_returns,
                "sharpe_ratio": mean_return / volatility if volatility > 0 else 0,
            }
        except KeyError:
            st.warning(f"Data for {ticker} is missing. Skipping.")
    return metrics

# Utility functions
def calculate_portfolio_metrics(selected_assets, weights):
    # Extract daily returns for the selected assets
    daily_returns = np.array([asset["daily_returns"] for asset in selected_assets]).T
    mean_returns = np.array([asset["mean_return"] for asset in selected_assets])

    # Compute the covariance matrix of daily returns
    covariance_matrix = np.cov(daily_returns, rowvar=False)

    # Portfolio mean as weighted average of individual means
    portfolio_mean_daily = np.dot(weights, mean_returns)

    # Portfolio volatility considering covariance
    portfolio_volatility_daily = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    # Convert daily metrics to monthly
    portfolio_mean_monthly = portfolio_mean_daily * 21  # Approx. 21 trading days per month
    portfolio_volatility_monthly = portfolio_volatility_daily * np.sqrt(21)

    return portfolio_mean_monthly, portfolio_volatility_monthly

# Monte Carlo Simulation using Geometric Brownian Motion (GBM)
def monte_carlo_simulation_gbm(mu, sigma, monthly_investment, time_horizon, num_simulations=10000):
    dt = 1 / 12  # Monthly steps
    results = []
    for _ in range(num_simulations):
        portfolio_value = 0
        for month in range(time_horizon * 12):
            monthly_return = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()) - 1
            portfolio_value += monthly_investment
            portfolio_value *= (1 + monthly_return)
        results.append(portfolio_value)
    return results

# Optimize portfolio allocation based on risk tolerance
def optimize_portfolio(metrics, risk_tolerance):
    stocks = sorted(metrics.items(), key=lambda x: x[1]["volatility"])
    bonds = sorted(metrics.items(), key=lambda x: x[1]["volatility"])

    if risk_tolerance == "Low":
        stock_picks = stocks[:20]  # Select 20 stocks with lowest volatility
        bond_picks = bonds[:5]  # Select 5 bonds with lowest volatility
    elif risk_tolerance == "Moderate":
        stock_picks = stocks[10:25]  # Select middle 15 stocks
        bond_picks = bonds[5:15]  # Select middle 10 bonds
    else:  # High risk tolerance
        stock_picks = stocks[-15:]  # Select top 15 stocks by volatility
        bond_picks = bonds[-5:]  # Select top 5 bonds by volatility

    # From selected stocks, choose top 5 based on Sharpe ratio
    top_stock_picks = sorted(stock_picks, key=lambda x: x[1]["sharpe_ratio"], reverse=True)[:5]
    return top_stock_picks, bond_picks

# Streamlit app
st.title("Optimized Investment Portfolio Decision Support System")

# Load data
file_path = 'financial_data_last_year.csv'  # Ensure this matches your uploaded file
try:
    data = load_data(file_path)
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    data = None

if data is not None:
    st.write("**Note:** This analysis is based on one year of historical data. Future performance may vary.")

    # User inputs
    monthly_investment = st.number_input("Amount willing to invest monthly ($)", min_value=0.0, step=100.0)
    age = st.slider("Enter your age", 18, 100)
    retirement_age = st.slider("Set your retirement age", min_value=50, max_value=100, value=65)
    adjust_to_retirement = st.button("Adjust Horizon to Retirement Age")

    if adjust_to_retirement:
        horizon = max(retirement_age - age, 1)
    else:
        horizon = st.slider("Set your investment horizon (years)", min_value=1, max_value=(100 - age), value=max(retirement_age - age, 1))

    st.write(f"Your investment horizon is set to {horizon} years.")

    risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])

    # Default stock-bond allocation based on risk tolerance
    if risk_tolerance == "Low":
        stock_weight_default = 0.6
        bond_weight_default = 0.4
    elif risk_tolerance == "Moderate":
        stock_weight_default = 0.7
        bond_weight_default = 0.3
    else:
        stock_weight_default = 0.8
        bond_weight_default = 0.2

    st.write("### Default Allocation")
    st.write(f"**Stocks Allocation (%):** {stock_weight_default * 100:.2f}%")
    st.write(f"**Bonds Allocation (%):** {bond_weight_default * 100:.2f}%")

    # Allow user to adjust allocation
    st.write("### Adjust Allocation")
    stock_weight = st.slider("Stocks Allocation (%)", 0.0, 100.0, stock_weight_default * 100.0) / 100.0
    bond_weight = 1 - stock_weight

    st.write(f"**Adjusted Stocks Allocation (%):** {stock_weight * 100:.2f}%")
    st.write(f"**Adjusted Bonds Allocation (%):** {bond_weight * 100:.2f}%")

    # Extract tickers dynamically from the dataset
    tickers = list(set([col.split(".")[0] for col in data.columns if "." in col]))

    # Extract metrics
    metrics = extract_metrics(data, tickers)

    if not metrics:
        st.error("No valid metrics available for analysis. Check your dataset.")
    else:
        # Optimize portfolio
        stock_picks, bond_picks = optimize_portfolio(metrics, risk_tolerance)

        # Calculate portfolio metrics
        selected_assets = [item[1] for item in stock_picks + bond_picks]
        weights = ([stock_weight / len(stock_picks)] * len(stock_picks)) + ([bond_weight / len(bond_picks)] * len(bond_picks))
        portfolio_mu, portfolio_sigma = calculate_portfolio_metrics(selected_assets, weights)

        # Display selected assets
        st.subheader("Selected Investments")
        st.write(f"**Stocks:** {', '.join([stock[0] for stock in stock_picks])}")
        st.write(f"**Bonds:** {', '.join([bond[0] for bond in bond_picks])}")

        # Display portfolio metrics
        st.subheader("Portfolio Metrics")
        st.write(f"**Portfolio Mean (mu) (Monthly):** {portfolio_mu:.6f}")
        st.write(f"**Portfolio Volatility (sigma) (Monthly):** {portfolio_sigma:.6f}")

        # Monte Carlo simulation
        st.subheader("Monte Carlo Simulation for Portfolio")
        progress = st.progress(0)
        simulations = monte_carlo_simulation_gbm(
            portfolio_mu, portfolio_sigma, monthly_investment, horizon
        )
        progress.progress(100)

        # Aggregate results
        total_invested = monthly_investment * 12 * horizon

        st.write(f"Total Amount Invested Over {horizon} Years: ${total_invested:,.2f}")
        st.write(f"5th Percentile: ${np.percentile(simulations, 5):,.2f}")
        st.write(f"95th Percentile: ${np.percentile(simulations, 95):,.2f}")

        # Plot results
        fig, ax = plt.subplots()
        ax.hist(simulations, bins=50, alpha=0.7)
        ax.set_title("Monte Carlo Simulation Results")
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Display portfolio allocation dynamically
        st.subheader("Portfolio Allocation")
        total_allocation = len(stock_picks) + len(bond_picks)
        st.write(f"**Stocks Allocation (%):** {stock_weight * 100:.2f}%")
        st.write(f"**Bonds Allocation (%):** {bond_weight * 100:.2f}%")
