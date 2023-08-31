# pylint: disable=missing-module-docstring
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Algorithms = ["mean_rev", "linear_model", "momentum"]


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
    combined_signals = reduce(lambda  left,right: pd.merge(
        left,right,on=['Date'],how='outer'), signals_dfs)
    # ToDo change merging so that counter columns are not merged and column names are meaningful
    combined_signals.drop(combined_signals.columns[[0,3,5]], axis=1, inplace=True)
    print(combined_signals)
    return combined_signals, [mean_rev_signals,model_signals,momentum_signals]

def strategy_implementation():
    """ 
    Define a simple trading strategy, defines the activty to make and identifes changes
    """
    # ToDo This function only uses one Signal atm
    generated_signals['Position'] = np.where(
        generated_signals['Signals'] == 1, 'Buy',
        np.where(generated_signals['Signals'] == -1, 'Sell', 'Hold'))
    generated_signals['Position'] = generated_signals['Position'].fillna(0)
    generated_signals['Trade'] = generated_signals['Signals'].diff()

    for dataframe in alg_signals.values():
        dataframe['Position'] = np.where(
            dataframe['Signals'] == 1, 'Buy',
            np.where(dataframe['Signals'] == -1, 'Sell', 'Hold'))
        dataframe['Position'] = dataframe['Position'].fillna(0)
        dataframe['Trade'] = dataframe['Signals'].diff()
    return generated_signals

def backtesting(signals_data):
    """ Simple backtesting function """
    initial_capital = 100000  # Initial capital in dollars
    signals_data['Portfolio'] = initial_capital + (signals_data['Trade'] * stock_data['Close'])
    signals_data['Portfolio'] = signals_data['Portfolio'].fillna(method='ffill')
    print(signals_data)

def gen_metrics(signals_data):
    """ Perfomance metrics """
    returns = signals_data['Portfolio'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    print(cumulative_returns)
    # Assuming 252 trading days in a year
    annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(signals_data)) - 1
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
    plt.plot(stock_data.index, metrics['cummulative returns'], label='Strategy')
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
    generated_signals, signals = signal_incoporation(TICKR_STR)
    alg_signals = {Algorithms: signals
                   for Algorithms, signals in zip(Algorithms, signals)}
    print(strategy_implementation())
    print(alg_signals)
    for model in alg_signals.values():
        backtesting(model)
        model_metrics = gen_metrics(model)
        show_data(model_metrics)
