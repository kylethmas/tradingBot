# pylint: disable=missing-module-docstring
from functools import reduce
import pandas as pd
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
    signal_to_position = {
        1: 'Buy',
        -1: 'Sell',
        0: 'Hold'
    }

    generated_signals['Position'] = generated_signals['Signals'].map(signal_to_position)
    generated_signals['Position'].fillna('Hold', inplace=True)
    generated_signals['Trade'] = generated_signals['Signals'].diff()

    for dataframe in alg_signals.values():
        dataframe['Position'] = dataframe['Signals'].map(signal_to_position)
        dataframe['Position'].fillna('Hold', inplace=True)
        dataframe['Trade'] = dataframe['Signals'].diff()

    return generated_signals

def backtesting(signals_data, initial_capital=100000):
    """
    Backtesting function for evaluating trading strategy performance.
    
    Parameters:
    - signals_data (pd.DataFrame): DataFrame containing trading signals and trade actions.
    - initial_capital (float): Initial capital in dollars.
    
    Returns:
    - pd.DataFrame: DataFrame with portfolio values after backtesting.
    """
    # Calculate portfolio values based on trading signals and actions
    signals_data['Portfolio'] = initial_capital + (signals_data['Trade'] * stock_data['Close'])

    # Fill missing portfolio values using forward fill
    signals_data['Portfolio'] = signals_data['Portfolio'].fillna(method='ffill')

    return signals_data

def gen_metrics(signals_data):
    """
    Calculate performance metrics for a trading strategy.
    
    Parameters:
    - signals_data (pd.DataFrame): DataFrame containing trading signals and portfolio values.
    - risk_free_rate (float): Risk-free rate for calculating Sharpe ratio (default: 0.03).
    
    Returns:
    - dict: A dictionary containing calculated performance metrics.
    """
    returns = signals_data['Portfolio'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    # Assuming 252 trading days in a year
    annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(signals_data)) - 1
    std_dev = returns.std()
    sharpe_ratio = (annualized_return - 0.03) / std_dev  # Assuming risk-free rate of 3%
    metrics = {
        'returns':returns,
        'cummulative_returns':cumulative_returns,
        'annualised_returns':annualized_return,
        'std_dev' : std_dev,
        'sharpe_ratio':sharpe_ratio
        }
    return metrics

def show_data(title,metrics):
    """
    Visualize performance metrics of a trading algorithm.
    
    Parameters:
    - metrics (dict): A dictionary containing performance metrics.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    plt.title(title)

    # Plot cumulative returns
    axes[0].plot(metrics['cummulative_returns'])
    axes[0].set_title('Cumulative Returns')

    # Display metrics as a table
    metrics_table = [
        ['Annualized Return', metrics['annualised_returns']],
        ['Standard Deviation', metrics['std_dev']],
        ['Sharpe Ratio', metrics['sharpe_ratio']]
    ]
    axes[1].axis('off')
    axes[1].table(cellText=metrics_table, loc='center')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Display the plots
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
    for name,model in zip(Algorithms, signals):
        backtesting(model)
        model_metrics = gen_metrics(model)
        show_data(name, model_metrics)
