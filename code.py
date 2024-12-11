import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Enhanced Utility functions
def return_utility(expected_return, weight, preference):
    factor = 1.5 if preference == "High" else 1.0 if preference == "Moderate" else 0.8
    return expected_return * weight * factor

def risk_utility(volatility, weight, preference):
    if volatility == 0:
        return 0
    risk_inverse = 1 / volatility
    factor = 0.8 if preference == "Low" else 1.0 if preference == "Moderate" else 1.2
    return risk_inverse * weight * factor

def horizon_utility(horizon, weight, preference):
    factor = 1.2 if preference == "High" else 1.0 if preference == "Moderate" else 0.8
    return horizon * weight * factor

# Monte Carlo Simulation
def monte_carlo_simulation(mean_return, volatility, monthly_investment, time_horizon, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        portfolio_value = 0
        for year in range(time_horizon):
            annual_return = np.random.normal(mean_return, volatility)
            for month in range(12):
                portfolio_value += monthly_investment
                portfolio_value *= (1 + (annual_return / 12))
        results.append(portfolio_value)
    return results

# Streamlit app
st.title("Investment Portfolio Decision Support System")

# Load data
file_path = 'financial_data_last_year.csv'  # Ensure this matches your uploaded file
try:
    data = load_data(file_path)
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    data = None

if data is not None:
    st.write("**Note:** This analysis is based on only one year of historical data. Future performance may differ significantly.")

    # User inputs
    income = st.number_input("Enter your monthly income ($)", min_value=0.0, step=100.0)
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

    # Assign weights
    weights = {
        "return": 0.5 if risk_tolerance == "High" else 0.3,
        "risk": 0.3 if risk_tolerance == "Moderate" else 0.2,
        "horizon": 0.2 if risk_tolerance == "Low" else 0.3,
    }

    # Extract tickers dynamically from the dataset
    tickers = list(set([col.split(".")[0] for col in data.columns if "." in col]))

    # Extract metrics
    metrics = extract_metrics(data, tickers)

    if not metrics:
        st.error("No valid metrics available for analysis. Check your dataset.")
    else:
        # Stock and bond selection based on risk tolerance and metrics
        stock_candidates = [item for item in metrics.items() if item[0] not in ["SPY"]]
        bond_candidates = [item for item in metrics.items() if item[0] in ["TLT", "BND", "IEF"]]  # Example bond tickers

        # Enhanced selection logic for diverse portfolios
        stock_picks = sorted(stock_candidates, key=lambda x: (return_utility(x[1]['mean_return'], weights['return'], risk_tolerance) - risk_utility(x[1]['volatility'], weights['risk'], risk_tolerance)), reverse=True)[:4]
        bond_picks = sorted(bond_candidates, key=lambda x: (return_utility(x[1]['mean_return'], weights['return'], risk_tolerance) - risk_utility(x[1]['volatility'], weights['risk'], risk_tolerance)), reverse=True)[:3]

        # Display top recommendations
        st.subheader("Top Recommendations")
        st.write(f"**Selected Stocks:** {', '.join([stock[0] for stock in stock_picks])}")
        st.write(f"**Selected Bonds:** {', '.join([bond[0] for bond in bond_picks])}")

        # Monte Carlo simulation for the portfolio
        st.subheader("Monte Carlo Simulation for Portfolio")
        total_simulations = []
        for ticker, metric in stock_picks + bond_picks:
            total_simulations.extend(monte_carlo_simulation(
                metric["mean_return"],
                metric["volatility"],
                monthly_investment / len(stock_picks + bond_picks),
                horizon,
            ))

        total_invested = monthly_investment * 12 * horizon
        st.write(f"Total Amount Invested Over {horizon} Years: ${total_invested:,.2f}")
        st.write(f"Simulated Portfolio Value after {horizon} years:")
        st.write(f"Mean: ${np.mean(total_simulations):,.2f}")
        st.write(f"5th Percentile: ${np.percentile(total_simulations, 5):,.2f}")
        st.write(f"95th Percentile: ${np.percentile(total_simulations, 95):,.2f}")

        # Plot simulation results
        fig, ax = plt.subplots()
        ax.hist(total_simulations, bins=50, alpha=0.7)
        ax.set_title("Monte Carlo Simulation Results")
        ax.set_xlabel("Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
