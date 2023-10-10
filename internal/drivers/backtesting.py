# pylint: disable=missing-module-docstring
import os
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
Algorithms = ["momentum", "mean_rev", "linear_model"]
TICKR_OUTPUT_PATH = 'tickrOutputs/'

def get_tickr():
    """
    Get ticker from input and checks data exists, raises value error if nof
    """
    ticker = input("Enter ticker:").upper()
    dir_list = os.listdir(TICKR_OUTPUT_PATH)
    if ticker not in dir_list:
        raise ValueError('inputed ticker does not exist, run signal_generator.py and try again')
    return ticker


def data_prep(ticker):
    """
    Load historical stock data
    """
    data = pd.read_csv(f'{TICKR_OUTPUT_PATH}{ticker}/{ticker}.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def signal_incoporation(ticker):
    """ Load generated signals """
    momentum_signals = pd.read_csv(
        f'{TICKR_OUTPUT_PATH}{ticker}/{ticker}_momentum_signals.csv')
    mean_rev_signals = pd.read_csv(
        f'{TICKR_OUTPUT_PATH}{ticker}/{ticker}_meanrev_signals.csv')
    model_signals = pd.read_csv(
        f'{TICKR_OUTPUT_PATH}{ticker}/{ticker}_model_signals.csv')
    signals_dfs = [momentum_signals, mean_rev_signals, model_signals]

    # Merge signals with historical data
    combined_signals = reduce(lambda  left,right: pd.merge(
        left,right,on=['Date'],how='outer'), signals_dfs)
    # ToDo change merging so that counter columns are not merged and column names are meaningful
    combined_signals.drop(combined_signals.columns[[0,3,5]], axis=1, inplace=True)
    print(combined_signals)
    return combined_signals, signals_dfs

def strategy_implementation(signals_df, alg_signals_df):
    """ 
    Define a simple trading strategy, defines the activty to make and identifes changes
    """
    # ToDo This function only uses one Signal atm
    signal_to_position = {
        1: 'Buy',
        -1: 'Sell',
        0: 'Hold'
    }

    signals_df['Position'] = signals_df['Signals'].map(signal_to_position)
    signals_df['Position'].fillna('Hold', inplace=True)
    signals_df['Trade'] = signals_df['Signals'].diff()

    for dataframe in alg_signals_df.values():
        dataframe['Position'] = dataframe['Signals'].map(signal_to_position)
        dataframe['Position'].fillna('Hold', inplace=True)
        dataframe['Trade'] = dataframe['Signals'].diff()

    print("Strategy Implementation")

    return signals_df

def backtesting(signals_data, stock_df, initial_capital=100000):
    """
    Backtesting function for evaluating trading strategy performance.

    Parameters:
    - signals_data (pd.DataFrame): DataFrame containing trading signals and trade actions.
    - initial_capital (float): Initial capital in dollars.

    Returns:
    - pd.DataFrame: DataFrame with portfolio values after backtesting.
    """
    # Initialize the portfolio value
    shares_owned = 0
    cash_held = initial_capital
    portfolio_value = 0

    # Loop over the rows of the signals dataframe
    for i in range(len(signals_data)):
        # Get the current signal and position
        signal = signals_data.loc[i, "Signals"]

        # Update the portfolio value
        if signal == 1:
            # Calculate the number of shares
            shares_owned += cash_held / stock_df.loc[i, "Close"]
            cash_held = 0
        elif signal == -1:
            cash_held += stock_df.loc[i, "Close"] * shares_owned
            shares_owned = 0
        else:
            pass

        portfolio_value = cash_held + shares_owned * stock_df.loc[i, "Close"]
        # Add the portfolio value to the signals_data DataFrame
        signals_data.loc[i, "Portfolio"] = portfolio_value

    # Fill missing portfolio values using forward fill
    signals_data['Portfolio'] = signals_data['Portfolio'].fillna(method='ffill')
    print(signals_data)

    # Return the trading strategy's returns dataframe.
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
    return plt

def save_data(ticker_name, alg_name, alg_model, metrics):
    """ Saves data """
    alg_model.to_csv(f'{TICKR_OUTPUT_PATH}{ticker_name}/{ticker_name}{alg_name}_model.csv')
    metricsdf = pd.DataFrame(metrics)
    metricsdf.to_csv(f'{TICKR_OUTPUT_PATH}{ticker_name}/{ticker_name}{alg_name}_metrics.csv')


# Step 7: Parameter Optimization (Optional)
# perform parameter optimization here by testing different parameter values.

# Step 8: Out-of-Sample Testing (Optional)
# Reserve a portion of the data for out-of-sample testing and repeat the steps.

# Step 9: Risk Management (Optional)
# Implement risk management techniques such as position sizing.

# Step 10: Documentation and Reporting
# Create a report summarizing the strategy, metrics, and results.





if __name__=="__main__":
    tickr_str = get_tickr()
    stock_data = data_prep(tickr_str)
    generated_signals, signals = signal_incoporation(tickr_str)
    alg_signals = {Algorithms: signals
                   for Algorithms, signals in zip(Algorithms, signals)}
    print(strategy_implementation(generated_signals, alg_signals))
    print(alg_signals)
    for name,model in zip(Algorithms, signals):
        print(backtesting(model, stock_data))
        model_metrics = gen_metrics(model)
        figure = show_data(name, model_metrics)
        save_data(tickr_str, name, model, model_metrics)
