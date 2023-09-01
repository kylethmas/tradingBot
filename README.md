Activate virtualenv
'''
source tradingBot/bin/activate
'''

To run signal genrator
'''
python internal/drivers/signal_generator.py
'''
And input a stock tickr which can be found from Yahoo Finance

To run backtesting
'''
python internal/drivers/backtesting.py
'''
Make sure to choose a stock tickr which is already in tickrOutputs


Current Algorithhms:
Mean reversion algorithm 
- assumption that an asset's price will tend to converge to the average price over time
- Mean rev = mean reversion algorithm

Momentum trading
- assumption that an asset's price will tend to converge to the average price over time
- if a stock is soaring after releasing a stellar earnings report, a momentum trader might try to buy shares and ride the stock's price higher

Algorithmic trading model
- Simple linear regression model


the Sharpe ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk
In this case a risk free asset is is you had originally invested in Apple stock
