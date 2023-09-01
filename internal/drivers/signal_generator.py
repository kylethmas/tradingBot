# pylint: disable=missing-module-docstring
import os
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn import linear_model

TICKR_OUTPUT_PATH = 'tickrOutputs/'

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

    os.makedirs(f'{TICKR_OUTPUT_PATH}{ticker_string}', exist_ok=True)

    ticker_hist.to_csv(f'{TICKR_OUTPUT_PATH}{ticker_string}/{ticker_string}.csv')

    plot_tickr_history(ticker_hist)

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
    dataframe = pd.read_csv(f'{TICKR_OUTPUT_PATH}{ticker}/{ticker}.csv')

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
    plt.title("Mean Reversion Algorithm using bolligner bands")
    plt.legend()
    plt.show()
    dataframe.to_csv(
        f"{TICKR_OUTPUT_PATH}{ticker}/{ticker}_meanrev_signals.csv",
        columns = ["Date","Signals"]
        )


def plot_tickr_history(data):
    """
    This function plots a tcikrs close price history.

    Args:
    data: A pandas DataFrame of historical prices.
    """

    fig, ax1 = plt.subplots()

    ax1.plot(data["Close"])

    ax1.set(xlabel="Date", ylabel="Price", title="Tickr performance")
    ax1.legend("Close")
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
    dataframe = pd.read_csv(f"{TICKR_OUTPUT_PATH}{ticker}/{ticker}.csv")

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
    dataframe.to_csv(
        f"{TICKR_OUTPUT_PATH}{ticker}/{ticker}_momentum_signals.csv",
        columns = ["Date","Signals"]
        )

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
    dataframe = pd.read_csv(f"{TICKR_OUTPUT_PATH}{ticker}/{ticker}.csv")


    # Use linear regression to fit a model to the data
    x_axis = dataframe[["Open", "High", "Low", "Close", "Volume"]]
    y_axis = dataframe["Close"]
    model = linear_model.LinearRegression().fit(x_axis, y_axis)


    # Generate trading signals
    signals = generate_model_signals(dataframe, model)
    dataframe["Signals"] = signals
    dataframe.to_csv(
        f"{TICKR_OUTPUT_PATH}{ticker}/{ticker}_model_signals.csv",
        columns = ["Date","Signals"]
        )


if __name__=="__main__":
    TICKR_STR = input("Enter tickr:")
    check_ticker(TICKR_STR)
    yf_ticker = get_historical_data(TICKR_STR)
    print(yf_ticker)
    momentum_trading(TICKR_STR)
    mean_reversion_alg(TICKR_STR)
    algo_trading(TICKR_STR)
