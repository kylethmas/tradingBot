#!/usr/bin/env python
# coding: utf-8


#import external pandas_datareader library with alias of web
#import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
import os


# Gets historical data for #nvda

def getHistoricalData(ticker):

    tickr = yf.Ticker(ticker)
    tickrHist = tickr.history(start="2020-01-01", end="2023-01-01")
    print(tickrHist.head())

    os.makedirs(f'tickrOutputs/{ticker}', exist_ok=True)

    tickrHist.to_csv(f'tickrOutputs/{ticker}/{ticker}.csv')
    return tickr


# Create a function to generate trading signals

#The Bollinger Bands are then calculated by subtracting the rolling mean from the stock's close price, 
#and dividing the result by two times the rolling standard deviation. 
#If the Bollinger Band value is greater than 1, the function generates a sell signal (-1); 
#if the Bollinger Band value is less than -1, the function generates a buy signal (1); 
#otherwise, the function generates a hold signal (0).

def generate_signals(data):
    signals = []
    for i in range(len(data)):
        if data["Bollinger Band"][i] > 1:
            signals.append(-1)
        elif data["Bollinger Band"][i] < -1:
            signals.append(1)
        else:
            signals.append(0)
            
    return signals

#This strategy is based on the idea that prices will eventually return to their mean, or average, value. 
#The bot will buy when the price is below the average and sell when the price is above the average.

def meanReversionAlg(ticker):
    # Load historical data into a dataframe
    try:
        df = pd.read_csv(f'tickrOutputs/{ticker}/{ticker}.csv')
    except:
        getHistoricalData(ticker)
        df = pd.read_csv(f'tickrOutputs/{ticker}/{ticker}.csv')

    # Calculate the rolling mean and standard deviation
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()

    # Create a new column for the Bollinger Band
    df["Bollinger Band"] = (df["Close"] - rolling_mean) / (2 * rolling_std)


    # Generate trading signals
    signals = generate_signals(df)
    df["Signals"] = signals

    # Plot the data
    plt.plot(df["Close"], label='close')
    plt.ylabel("Price")
    plt.xlabel("Day")
    plt.plot(rolling_mean, label='rolling mean')
    plt.plot(rolling_mean + 2 * rolling_std, label='rolling mean + 2')
    plt.plot(rolling_mean - 2 * rolling_std, label='rolling mean - 2')
    plt.legend()
    plt.show()
    df["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_meanrev_signals.csv")
    

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

    plt.show()
    

# Create a function to generate trading signals

#If RSI value is greater than 70, the function generates a sell signal (-1); 
#if RSI value is less than 30, the function generates a buy signal (1); 
#otherwise, the function generates a hold signal (0).

def generate_RSI_signals(data):
    signals = []
    for i in range(len(data)):
        if data['RSI'][i] > 70:
            signals.append(-1)
        elif data['RSI'][i] < 30:
            signals.append(1)
        else:
            signals.append(0)
    return signals

#RSI is a momentum indicator that compares the magnitude of recent gains to recent 
#losses in order to determine overbought and oversold conditions of an asset.

def momentumTrading(ticker):
    # Load historical data into a dataframe
    try:
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")
    except:
        getHistoricalData(ticker)
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")

    # Create a new column for the 14-day relative strength index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi

    # Generate trading signals
    signals = generate_RSI_signals(df)
    df['Signals'] = signals
    
    df["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_momentum_signals.csv")
    
    plot_momentum_signals(df)
    
"""     correctness = 0 # if profit is positive we're making money !!!
    x=2
    print(df[x])
    for x in range(len(df)):
        print(x)
        if df[x-1]["Close"] < df[x]["Close"]:#if yesterday close < today close
            if signal[x] == -1: #if we sold 
                correctness += 1
            else:
                correctness -= 1
        else:
            if signal[x] == 1:
                correctness += 1
            else:
                correctness -= 1
    print(correctness) """

# Create a function to generate trading signals
def generate_model_signals(data, model):
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
    try:
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")
    except:
        getHistoricalData(ticker)
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")


    # Use linear regression to fit a model to the data
    x = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Close"]
    model = LinearRegression().fit(x, y)


    # Generate trading signals
    signals = generate_model_signals(df, model)
    df["Signals"] = signals
    df["Signals"].to_csv(f"tickrOutputs/{ticker}/{ticker}_model_signals.csv")
    
    

def test(ticker):
    # Load historical data into a dataframe
    try:
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")
    except:
        getHistoricalData(ticker)
        df = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}.csv")

    # Initialize variables for the strategy
    initial_capital = 100000
    positions = pd.read_csv(f"tickrOutputs/{ticker}/{ticker}_meanrev_signals.csv",usecols="Signals")
    cash = []
    
    print(positions)

    # Iterate through the data and execute trades
    for i in range(len(df)):
        if i == 0:
            cash.append(initial_capital)
        else:

            # Execute trades
            if positions[i] == 1:
                cash.append(cash[i-1] - df['Open'][i])
            elif positions[i] == -1:
                cash.append(cash[i-1] + df['Open'][i])
            else:
                cash.append(cash[i-1])

    # Add the cash and position values to the dataframe
    df['Positions'] = positions
    df['Cash'] = cash

    # Calculate the total value of the portfolio
    df['Total'] = df['Cash'] + df['Open'] * df['Positions']

    # Plot the portfolio value over time
    plt.plot(df['Total'])
    plt.show()

if __name__=="__main__":
    tickr = input("Enter tickr:")
    getHistoricalData(tickr)
    meanReversionAlg(tickr)
    momentumTrading(tickr)
    algo_trading(tickr)
