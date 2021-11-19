# import cryptocompare -> didn't work, will try later again
import math

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored as cl

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 8)
# Create FetchDataCryptoCompare for data per seconds
"""
--> Determine the dataset
    - choose the cryptos to work on. Maybe build some portfolios of cryptos later on.
    - data in days, hours, seconds ? Which one is the most reliable to forecast the coming seconds, hours or days
"""


class FetchDataYahoo:

    # the portfolio will be as class argument
    PORTFOLIO = 'BTC-USD'  # ETH-USD DOGE-USD LTC-USD DOT1-USD ADA-USD'

    # You have to put as parameter of the class the cryptos you want to see
    def __init__(self):
        self.start = '2020-02-25'
        self.end = '2021-02-25'

    # this is the training set on which we will perform our analysis
    def training_set(self):
        return yf.download(self.PORTFOLIO, start=self.start, end=self.end).ffill().dropna()

    # this is the testing data set. We will compare it with the forecast we get from our model.
    def test_set(self, n_days=15):
        end_test = datetime.datetime.strptime(self.end, "%Y-%m-%d") + datetime.timedelta(days=n_days)
        return yf.download(self.PORTFOLIO, start=self.end, end=end_test).ffill().dropna()


"""
--> Run statistics:
    - calculate basic stats : annualized returns & volatility,  max drawdown, calmar ratio (ann. returns/max drawdown), 
      ann. Sharpe ratio ((Rp-Rf)/std), correlation between assets, heatmap and clustermap (seaborn package)
    - compare results of a regular SMA trading strategy and the WML strategy (wml overweights recent prices)
    - Try the same on volumes instead of prices (creation of a technical indicator) = VOLUME-WEIGHTED MOVING AVERAGE
"""


class BasicStats(FetchDataYahoo):

    def __init__(self):
        super().__init__()
        self.data = self.training_set()
        self.prices = self.training_set()["Close"]
        self.volume = self.training_set()["Volume"]

    # get the returns
    def returns_func(self):
        return self.prices.pct_change()

    # annualized the returns
    def annualized_returns(self):
        return round(self.returns_func().mean()*252, 2)

    # annualized volatility
    def annualized_volatility(self):
        return round(self.returns_func().std()*252**0.5, 2)

    # get the maximum drawdown of each assets
    def max_drawdown(self):
        return round((self.prices/(self.prices.cummax())-1.0).min(), 2)

    # calmar ratio
    def calmar_ratio(self):
        """ The higher the Calmar ratio, the better it performed on a risk-adjusted basis during the given period """
        return round(-1 * (self.annualized_returns() / self.max_drawdown()), 2)

    def annualized_sharpe_ratio(self):
        """
        The Sharpe ratio shows whether the portfolio's excess returns are due to smart investment decisions or a
        result of taking a higher risk. The higher a portfolio's Sharpe ratio, the better its risk-adjusted performance.
        """
        rfr = 0  # it is commonly admitted that the current risk-free rate is 0
        ann_sharpe = ((self.annualized_returns()-rfr) / self.annualized_volatility())
        return ann_sharpe  # .apply(lambda x: x*(x > 0))

    # calculates the correlation between the assets of the portfolio
    def correlation(self):
        try:
            return self.returns_func().corr()
        except TypeError:
            print("Correlation can't be calculated over a single asset!")

    # plotting the correlation heatmap
    def heatmap(self):
        hm = sns.heatmap(self.correlation())
        hm.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=20)
        plt.show()

    # plotting the clustermap
    def clustermap(self):
        sns.clustermap(self.correlation()) #, vmin=-1, vmax=1, annot=True)
        plt.title('Correlation cluster map', loc='center', fontdict={'fontsize': 20}, pad=20)
        plt.show()

    # Simple Moving Average for eacg asset of a portfolio
    def SMA(self, criteria='Close', slow=50, fast=20):  # the criteria can also be 'Volume'

        if type(self.data[criteria]) == pd.Series:  # we need to do it differently when only one crypto is analyzed
            slow_table = pd.DataFrame(self.data[criteria].rolling(window=slow).mean()).rename(
                columns={'Close': criteria + '_slowSMA'})
            fast_table = pd.DataFrame(self.data[criteria].rolling(window=fast).mean()).rename(
                columns={'Close': criteria + '_fastSMA'})
            return pd.concat([slow_table, fast_table], join='inner', axis=1)

        slow_table = self.data[criteria].rolling(window=slow).mean()
        fast_table = self.data[criteria].rolling(window=fast).mean()

        slow_col = []
        for col_name in slow_table.columns:
            slow_col.append(col_name + ' ' + criteria + '_slowSMA')

        fast_col = []
        for col_name in fast_table.columns:
            fast_col.append(col_name + ' ' + criteria + '_slowSMA')

        slow_table.columns = slow_col
        fast_table.columns = fast_col

        return pd.concat([slow_table, fast_table], join='inner', axis=1)

    # Exponential Moving Average
    def EMA(self, criteria='Close', slow=14, fast=3):  # criteria can be 'Volume' also

        if type(self.data[criteria]) == pd.Series:  # we need to do it differently when only one crypto is analyzed
            slow_table = pd.DataFrame(self.data[criteria].ewm(span=slow, adjust=False).mean()).rename(
                columns={'Close': criteria + '_slowEMA'})
            fast_table = pd.DataFrame(self.data[criteria].ewm(span=fast, adjust=False).mean()).rename(
                columns={'Close': criteria + '_fastEMA'})
            return pd.concat([slow_table, fast_table], join='inner', axis=1)

        slow_table = self.data[criteria].ewm(span=slow, adjust=False).mean()
        fast_table = self.data[criteria].ewm(span=fast, adjust=False).mean()

        slow_col = []
        for col_name in slow_table.columns:
            slow_col.append(col_name + ' ' + criteria + '_slowEMA')

        fast_col = []
        for col_name in fast_table.columns:
            fast_col.append(col_name + ' ' + criteria + '_slowEMA')

        slow_table.columns = slow_col
        fast_table.columns = fast_col

        return pd.concat([slow_table, fast_table], join='inner', axis=1)


# For now, we can do it only for one asset.
# We have to build a portfolio (e.g. capitalization-weighted) and calculate its level
# to be able to apply it to this class and see the result.
class StrategyCalculator(BasicStats):

    def __init__(self, strategy: str = 'SMA', **kwargs):
        self.__dict__.update({'params': kwargs})
        super().__init__()
        self.strategy = None
        try:
            self.strategy = getattr(self, strategy)
        except AttributeError as e:
            raise e

    # IF SMA(SHORT PERIOD) > SMA(LONG PERIOD) => BUY
    # IF SMA(LONG PERIOD) > SMA(SHORT PERIOD) => SELL
    def strategy_implementation(self):
        data = self.prices
        strat = self.strategy()

        slow = strat.iloc[:, 0]  # return first column: slow
        fast = strat.iloc[:, 1]  # return first column: fast
        buy_price = []
        sell_price = []
        ma_signal = []
        signal = 0

        for i in range(len(data)):
            if fast[i] > slow[i]:
                if signal != 1:
                    buy_price.append(data[i])
                    sell_price.append(np.nan)
                    signal = 1
                    ma_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    ma_signal.append(0)
            elif slow[i] > fast[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(data[i])
                    signal = -1
                    ma_signal.append(-1)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    ma_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                ma_signal.append(0)

        return buy_price, sell_price, ma_signal

    def trading_signals(self):
        strat = self.strategy()
        slow = strat.iloc[:, 0]  # return first column: slow
        fast = strat.iloc[:, 1]  # return first column: fast

        buy_price, sell_price = self.strategy_implementation()[0], self.strategy_implementation()[1]

        plt.plot(self.prices, alpha=0.3, label=self.PORTFOLIO)
        plt.plot(fast, alpha=0.6, label='SMA 20')
        plt.plot(slow, alpha=0.6, label='SMA 50')
        plt.scatter(self.data.index, buy_price, marker='^', s=200, color='darkblue', label='BUY SIGNAL')
        plt.scatter(self.data.index, sell_price, marker='v', s=200, color='crimson', label='SELL SIGNAL')
        plt.legend(loc='upper left')
        plt.title('STRATEGY TRADING SIGNALS')
        plt.show()

    def positions(self):
        strat = self.strategy()
        signal = self.strategy_implementation()[2]

        position = []
        for i in range(len(signal)):
            if signal[i] > 1:
                position.append(0)
            else:
                position.append(1)

        for i in range(len(self.prices)):
            if signal[i] == 1:
                position[i] = 1
            elif signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1]

        buy_price, sell_price = self.strategy_implementation()[1], self.strategy_implementation()[2]

        buy_price = pd.DataFrame(buy_price).rename(columns={0: 'buy_price'}).set_index(self.data.index)
        sell_price = pd.DataFrame(sell_price).rename(columns={0: 'sell_price'}).set_index(self.data.index)
        signal = pd.DataFrame(signal).rename(columns={0: 'ma_signal'}).set_index(self.data.index)
        position = pd.DataFrame(position).rename(columns={0: 'ma_position'}).set_index(self.data.index)

        frames = [strat, buy_price, sell_price, signal, position]
        strat_pos = pd.concat(frames, join='inner', axis=1)
        strat_pos = strat_pos.reset_index().drop('Date', axis=1)

        return strat_pos

    def backtest(self):
        ret = pd.DataFrame(np.diff(self.prices)).rename(columns={0: 'returns'})
        pos = self.positions()
        ma_strategy_ret = []

        for i in range(len(ret)):
            try:
                returns = ret['returns'][i] * pos['ma_position'][i]
                ma_strategy_ret.append(returns)
            except:
                pass

        ma_strategy_ret_df = pd.DataFrame(ma_strategy_ret).rename(columns={0: 'ma_returns'})

        investment_value = 100000
        number_of_stocks = math.floor(investment_value / self.prices[1])
        sma_investment_ret = []

        for i in range(len(ma_strategy_ret_df['ma_returns'])):
            returns = number_of_stocks * ma_strategy_ret_df['ma_returns'][i]
            sma_investment_ret.append(returns)

        ma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns={0: 'investment_returns'})
        total_investment_ret = round(sum(ma_investment_ret_df['investment_returns']), 2)
        print(cl('Profit gained from the strategy by investing $100K : ${} in 1 Year'.format(
                 total_investment_ret), attrs=['bold']))


if __name__ == '__main__':
    # print(FetchDataYahoo().training_set())
    # BasicStats().clustermap()
    # print(BasicStats().SMA("Volume"))
    # print(StrategyCalculator().trading_signals())
    # print(StrategyCalculator().positions())
    StrategyCalculator('EMA', criteria='Close').trading_signals()
