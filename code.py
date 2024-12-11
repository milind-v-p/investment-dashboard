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

# Utility functions
def return_utility(expected_return, weight):
    return expected_return * weight

def risk_utility(volatility, weight, preference="Moderate"):
    if volatility == 0:
        return 0
    risk_inverse = 1 / volatility
    if preference == "Low":
        return risk_inverse * weight * 1.2
    elif preference == "High":
        return risk_inverse * weight * 0.8
    return risk_inverse * weight

def horizon_utility(horizon, weight):
    return horizon * weight

# Monte Carlo Simulation
def monte_carlo_simulation(mean_return, volatility, monthly_investment, time_horizon, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        portfolio_value = 0
        annual_returns = np.random.normal(mean_return, volatility * 1.5 if volatility > 0 and risk_tolerance == "High" else (volatility * 0.5 if risk_tolerance == "Low" else volatility), time_horizon)
        for year in range(time_horizon):
            for month in range(12):
                portfolio_value += monthly_investment
                portfolio_value *= (1 + (annual_returns[year] / 12))
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
    # User inputs
    income = st.number_input("Enter your monthly income ($)", min_value=0.0, step=100.0)
    monthly_investment = st.number_input("Amount willing to invest monthly ($)", min_value=0.0, step=100.0)
    age = st.slider("Enter your age", 18, 100)
    retirement_age = 65
    horizon = max(retirement_age - age, 1)  # Adjust horizon based on retirement age
    st.write(f"Your investment horizon is automatically set to {horizon} years (until age {retirement_age}).")
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
        # Filter based on risk tolerance
        if risk_tolerance == "High":
            selected_tickers = sorted(metrics.items(), key=lambda x: x[1]['mean_return'], reverse=True)
        elif risk_tolerance == "Moderate":
            selected_tickers = sorted(metrics.items(), key=lambda x: x[1]['volatility'])
        else:  # Low risk
            selected_tickers = sorted(metrics.items(), key=lambda x: x[1]['volatility'])

        stock_picks = [ticker for ticker, _ in selected_tickers if ticker != "SPY"][:2] + ["SPY"]  # Include SPY as a stock
        bond_picks = [ticker for ticker, _ in selected_tickers if ticker not in stock_picks][:2]  # Select bonds not in stocks

        # Ensure exactly 3 stocks and 2 bonds
        stock_picks = stock_picks[:3]
        bond_picks = bond_picks[:2]

        # Display top recommendations
        st.subheader("Top Recommendations")
        st.write(f"**Selected Stocks:** {', '.join(stock_picks)}")
        st.write(f"**Selected Bonds:** {', '.join(bond_picks)}")

        # Monte Carlo simulation for the portfolio
        st.subheader("Monte Carlo Simulation for Portfolio")
        total_simulations = []
        for ticker in stock_picks + bond_picks:
            total_simulations.extend(monte_carlo_simulation(
                metrics[ticker]["mean_return"],
                metrics[ticker]["volatility"],
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
