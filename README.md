# Investment Portfolio Optimization Tool

This project is a **Streamlit-based application** designed to help users optimize their investment portfolios. By analyzing historical financial data, the app computes metrics like mean return, volatility, Sharpe ratio, and utility scores. It uses Monte Carlo simulations to model portfolio performance over a user-defined investment horizon.

## Key Features
- **Data Analysis**: Extracts financial metrics (mean return, volatility, Sharpe ratio) from a dataset.
- **Utility-Based Optimization**: Calculates utility scores based on risk tolerance and investment horizon.
- **Monte Carlo Simulation**: Models portfolio performance over time.
- **Interactive UI**: Allows users to input preferences like monthly investment, risk tolerance, and horizon.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/milind-v-p/investment-dashboard.git
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib
4. Run code.py:
   ```bash
   python code.py
5. Open Browser and put the following website in:
    ```bash
    http://localhost:8501

## Improvements for Future
- Incorporate live data from APIs for real-time insights.
- Add support for additional asset classes and advanced optimization techniques.
- Enhance visualizations and provide detailed insights into Monte Carlo simulation results.
- Allow users to save and compare multiple portfolio configurations.
