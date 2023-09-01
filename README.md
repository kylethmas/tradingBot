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
