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
def monte_carlo_simulation(mean_return, volatility, initial_investment, time_horizon, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        annual_returns = np.random.normal(mean_return, volatility, time_horizon)
        portfolio_value = initial_investment
        for annual_return in annual_returns:
            portfolio_value *= (1 + annual_return)
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
    amount = st.number_input("Amount willing to invest ($)", min_value=0.0, step=100.0)
    age = st.slider("Enter your age", 18, 100)
    horizon = st.slider("Investment horizon (years)", 1, 50)
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
        # Calculate utility scores
        recommendations = []
        for ticker, metric in metrics.items():
            r_utility = return_utility(metric["mean_return"], weights["return"])
            s_utility = risk_utility(metric["volatility"], weights["risk"], risk_tolerance)
            h_utility = horizon_utility(horizon, weights["horizon"])
            total_score = r_utility + s_utility + h_utility
            recommendations.append({"ticker": ticker, "score": total_score})

        # Sort recommendations
        sorted_recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)

        # Display top recommendations
        st.subheader("Top Recommendations")
        st.write("**Top 3 Investments:**")
        for rec in sorted_recommendations[:3]:
            st.write(f"- {rec['ticker']} (Utility Score: {rec['score']:.2f})")

        # Monte Carlo simulation for the top recommendation
        if sorted_recommendations:
            top_pick = sorted_recommendations[0]
            st.subheader(f"Monte Carlo Simulation for {top_pick['ticker']}")
            simulations = monte_carlo_simulation(
                metrics[top_pick["ticker"]]["mean_return"],
                metrics[top_pick["ticker"]]["volatility"],
                amount,
                horizon,
            )
            st.write(f"Simulated Portfolio Value after {horizon} years:")
            st.write(f"Mean: ${np.mean(simulations):,.2f}")
            st.write(f"5th Percentile: ${np.percentile(simulations, 5):,.2f}")
            st.write(f"95th Percentile: ${np.percentile(simulations, 95):,.2f}")

            # Plot simulation results
            fig, ax = plt.subplots()
            ax.hist(simulations, bins=50, alpha=0.7)
            ax.set_title("Monte Carlo Simulation Results")
            ax.set_xlabel("Portfolio Value ($)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No valid recommendations available.")
