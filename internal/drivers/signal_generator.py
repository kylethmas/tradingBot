# pylint: disable=missing-module-docstring
import os
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn import linear_model

def check_ticker(ticker):
    """
    Check ticker
    """
    try:
        return yf.Ticker(ticker).info['currentPrice']
    except ValueError as err:
        raise ValueError('Ticker not recognized') from err


def get_historical_data(ticker_string):
    """
    Download tickr history as CSV
    takes the ticker symbol as input
    """

    ticker = yf.Ticker(ticker_string)
    ticker_hist = ticker.history(start="2020-01-01", end="2023-01-01")
    print(ticker_hist.head())

    os.makedirs(f'tickrOutputs/{ticker_string}', exist_ok=True)

    ticker_hist.to_csv(f'tickrOutputs/{ticker_string}/{ticker_string}.csv')
    return ticker



def generate_signals(data):
    """
    Create a function to generate trading signals

    The Bollinger Bands are then calculated by subtracting the rolling mean from the 
    stock's close price, and dividing the result by two times the rolling standard deviation.
    If the Bollinger Band value is greater than 1, the function generates a sell signal (-1);
    if the Bollinger Band value is less than -1, the function generates a buy signal (1);
    otherwise, the function generates a hold signal (0).
    """
    signals = []
    for i in range(len(data)):
        if data["Bollinger Band"][i] > 1:
            signals.append(-1)
        elif data["Bollinger Band"][i] < -1:
            signals.append(1)
        else:
            signals.append(0)
    return signals


def mean_reversion_alg(ticker):
    """
    This strategy is based on the idea that prices will eventually 
    return to their mean, or average, value. 
    """
    # Load historical data into a dataframe
    dataframe = pd.read_csv(f'tickrOutputs/{ticker}/{ticker}.csv')

    # Calculate the rolling mean and standard deviation
    rolling_mean = dataframe["Close"].rolling(window=20).mean()
    rolling_std = dataframe["Close"].rolling(window=20).std()

    # Create a new column for the Bollinger Band
    dataframe["Bollinger Band"] = (dataframe["Close"] - rolling_mean) / (2 * rolling_std)


    # Generate trading signals
    signals = generate_signals(dataframe)
    dataframe["Signals"] = signals

    # Plot the data
    plt.plot(dataframe["Close"], label='close')
    plt.ylabel("Price")
    plt.xlabel("Day")
    plt.plot(rolling_mean, label='rolling mean')
    plt.plot(rolling_mean + 2 * rolling_std, label='rolling mean + 2')
    plt.plot(rolling_mean - 2 * rolling_std, label='rolling mean - 2')
    plt.legend()
    plt.show()
    dataframe["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_meanrev_signals.csv")


def plot_momentum_signals(data):
    """
    This function plots the signals compared to the actual data.

    Args:
    data: A pandas DataFrame of historical prices.
    signals: A pandas DataFrame of momentum trading signals.
    """

    signals = data["Signals"]

    fig, ax1 = plt.subplots()

    ax1.plot(data["Close"])
    ax1.plot(signals, color="red")

    ax1.set(xlabel="Date", ylabel="Price", title="Momentum Trading Signals")
    ax1.legend(["Close", "Signals"])
    fig.show()
    plt.show()


def generate_rsi_signals(data):
    """
    Create a function to generate trading signals
    If RSI value is greater than 70, the function generates a sell signal (-1);
    if RSI value is less than 30, the function generates a buy signal (1);
    otherwise, the function generates a hold signal (0).
    """

    signals = []
    for i in range(len(data)):
        if data['RSI'][i] > 70:
            signals.append(-1)
        elif data['RSI'][i] < 30:
            signals.append(1)
        else:
            signals.append(0)
    return signals


def momentum_trading(ticker):
    """
    RSI is a momentum indicator that compares the magnitude of recent gains to recent
    losses in order to determine overbought and oversold conditions of an asset.
    """
    # Load historical data into a dataframe
    dataframe = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")

    # Create a new column for the 14-day relative strength index (RSI)
    delta = dataframe['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    result = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + result))
    dataframe['RSI'] = rsi

    # Generate trading signals
    signals = generate_rsi_signals(dataframe)
    dataframe['Signals'] = signals
    dataframe["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_momentum_signals.csv")
    plot_momentum_signals(dataframe)

def generate_model_signals(data, model):
    """
    a function to generate trading signals
    """
    signals = []
    for i in range(len(data)):
        if i < len(data) - 1:
            prediction = model.predict(data[i:i+1][["Open", "High", "Low", "Close", "Volume"]])
            if prediction > data["Close"][i]:
                signals.append(1)
            else:
                signals.append(-1)
        else:
            signals.append(0)
    return signals

def algo_trading(ticker):
    """
    Creating an algorithmic trading function usig a Linear regression model
    """
    dataframe = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")


    # Use linear regression to fit a model to the data
    x_axis = dataframe[["Open", "High", "Low", "Close", "Volume"]]
    y_axis = dataframe["Close"]
    model = linear_model.LinearRegression().fit(x_axis, y_axis)


    # Generate trading signals
    signals = generate_model_signals(dataframe, model)
    dataframe["Signals"] = signals
    dataframe["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_model_signals.csv")


def test(ticker):
    """
    Functioanlity not working
    """

    # Load historical data into a dataframe
    dataframe = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")


    # Initialize variables for the strategy
    initial_capital = 100000
    positions = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}_meanrev_signals.csv",usecols="Signals")
    cash = []

    print(positions)

    # Iterate through the data and execute trades
    for i in range(len(dataframe)):
        if i == 0:
            cash.append(initial_capital)
        else:

            # Execute trades
            if positions[i] == 1:
                cash.append(cash[i-1] - dataframe['Open'][i])
            elif positions[i] == -1:
                cash.append(cash[i-1] + dataframe['Open'][i])
            else:
                cash.append(cash[i-1])

    # Add the cash and position values to the dataframe
    dataframe['Positions'] = positions
    dataframe['Cash'] = cash

    # Calculate the total value of the portfolio
    dataframe['Total'] = dataframe['Cash'] + dataframe['Open'] * dataframe['Positions']

    # Plot the portfolio value over time
    plt.plot(dataframe['Total'])
    plt.show()

if __name__=="__main__":
    ticker_str = input("Enter tickr:")
    check_ticker(ticker_str)
    yf_ticker = get_historical_data(ticker_str)
    print(yf_ticker)
    mean_reversion_alg(ticker_str)
    momentum_trading(ticker_str)
    algo_trading(ticker_str)
