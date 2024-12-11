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
            }
        except KeyError:
            st.warning(f"Data for {ticker} is missing. Skipping.")
    return metrics

# Utility functions
def calculate_portfolio_metrics(selected_assets):
    mean_returns = [asset["mean_return"] for asset in selected_assets]
    volatilities = [asset["volatility"]**2 for asset in selected_assets]
    portfolio_mean = np.mean(mean_returns)
    portfolio_volatility = np.sqrt(np.mean(volatilities))
    return portfolio_mean, portfolio_volatility

# Monte Carlo Simulation using Geometric Brownian Motion (GBM)
def monte_carlo_simulation_gbm(mu, sigma, monthly_investment, time_horizon, num_simulations=1000):
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

    # Assign weights dynamically based on risk tolerance
    weights = {
        "return": 0.6 if risk_tolerance == "High" else 0.4,
        "risk": 0.3 if risk_tolerance == "Moderate" else 0.2,
        "horizon": 0.1 if risk_tolerance == "Low" else 0.3,
    }

    # Extract tickers dynamically from the dataset
    tickers = list(set([col.split(".")[0] for col in data.columns if "." in col]))

    # Extract metrics
    metrics = extract_metrics(data, tickers)

    if not metrics:
        st.error("No valid metrics available for analysis. Check your dataset.")
    else:
        # Stock and bond selection based on utility scores
        stock_candidates = [item for item in metrics.items() if item[0] not in ["TLT", "BND", "IEF"]]
        bond_candidates = [item for item in metrics.items() if item[0] in ["TLT", "BND", "IEF"]]

        stock_picks = sorted(
            stock_candidates,
            key=lambda x: x[1]["mean_return"] - 0.5 * x[1]["volatility"],
            reverse=True
        )[:4]
        bond_picks = sorted(
            bond_candidates,
            key=lambda x: x[1]["mean_return"] - 0.2 * x[1]["volatility"],
            reverse=True
        )[:3]

        # Calculate portfolio metrics
        selected_assets = [item[1] for item in stock_picks + bond_picks]
        portfolio_mu, portfolio_sigma = calculate_portfolio_metrics(selected_assets)

        # Display selected assets
        st.subheader("Selected Investments")
        st.write(f"**Stocks:** {', '.join([stock[0] for stock in stock_picks])}")
        st.write(f"**Bonds:** {', '.join([bond[0] for bond in bond_picks])}")

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
        st.write(f"Simulated Portfolio Value after {horizon} years:")
        st.write(f"Mean: ${np.mean(simulations):,.2f}")
        st.write(f"5th Percentile: ${np.percentile(simulations, 5):,.2f}")
        st.write(f"95th Percentile: ${np.percentile(simulations, 95):,.2f}")

        # Plot results
        fig, ax = plt.subplots()
        ax.hist(simulations, bins=50, alpha=0.7)
        ax.set_title("Monte Carlo Simulation Results")
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
