# pylint: disable=missing-module-docstring
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def data_prep(ticker):
    """
    # Load historical stock data
    """
    data = pd.read_csv(f'internal/drivers/tickrOutputs/{ticker}/{ticker}.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def signal_incoporation(ticker):
    """ Load generated signals """
    mean_rev_signals = pd.read_csv(
        f'internal/drivers/tickrOutputs/{ticker}/{ticker}_meanrev_signals.csv')
    model_signals = pd.read_csv(
        f'internal/drivers/tickrOutputs/{ticker}/{ticker}_model_signals.csv')
    momentum_signals = pd.read_csv(
        f'internal/drivers/tickrOutputs/{ticker}/{ticker}_momentum_signals.csv')
    signals_dfs = [mean_rev_signals, model_signals, momentum_signals]

    # Merge signals with historical data
    signals = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],how='outer'), signals_dfs)
    # ToDo change merging so that counter columns are not merged and column names are meaningful
    signals.drop(signals.columns[[0,3,5]], axis=1, inplace=True)
    print(signals)
    return signals

def strategy_implementation():
    """ 
    Define a simple trading strategy 
    TO BE USED AS A HELPER FUNCTION 
    """
    stock_data['Position'] = np.where(
        stock_data['Signal'] == 'Buy', 1,
        np.where(stock_data['Signal'] == 'Sell', -1, 0))
    stock_data['Position'] = stock_data['Position'].fillna(0)
    stock_data['Trade'] = stock_data['Position'].diff()

def backtesting():
    """ Simple backtesting function """
    initial_capital = 100000  # Initial capital in dollars
    stock_data['Portfolio'] = initial_capital + (stock_data['Trade'] * stock_data['Close'])
    stock_data['Portfolio'] = stock_data['Portfolio'].fillna(method='ffill')

def gen_metrics():
    """ Perfomance metrics """
    returns = stock_data['Portfolio'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    # Assuming 252 trading days in a year
    annualized_return = (cumulative_returns[-1]) ** (252 / len(stock_data)) - 1
    std_dev = returns.std()
    sharpe_ratio = (annualized_return - 0.03) / std_dev  # Assuming risk-free rate of 3%
    metrics = {
        'returns':returns,
        'cummulative returns':cumulative_returns,
        'annualised returns':annualized_return,
        'standard deviation' : std_dev,
        'share ratio':sharpe_ratio
        }
    return metrics

def show_data(metrics):
    """ Step 6: Visualization """

    for key,value in metrics.items():
        print(key, ':', value)

    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, metrics['cumulative returns'], label='Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.legend()
    plt.show()

# Step 7: Parameter Optimization (Optional)
# perform parameter optimization here by testing different parameter values.

# Step 8: Out-of-Sample Testing (Optional)
# Reserve a portion of the data for out-of-sample testing and repeat the steps.

# Step 9: Risk Management (Optional)
# Implement risk management techniques such as position sizing.

# Step 10: Documentation and Reporting
# Create a report summarizing the strategy, metrics, and results.

if __name__=="__main__":
    TICKR_STR = 'AAPL'
    stock_data = data_prep(TICKR_STR)
    signal_incoporation(TICKR_STR)
