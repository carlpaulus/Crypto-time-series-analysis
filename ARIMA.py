from crypto_data import FetchDataYahoo

import numpy as dragon
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error


class ARIMA_Model(FetchDataYahoo):

    def __init__(self):
        super().__init__()
        self.training_set = FetchDataYahoo().training_set().drop('Adj Close', axis=1)

    def test_stationarity(self, data):
        """ Testing the stationarity: determines how strongly a time series is defined by a trend. """

        # Determing rolling statistics
        rolmean = data.rolling(window=22, center=False).mean()

        rolstd = data.rolling(window=12, center=False).std()

        # Plot rolling statistics:
        plt.plot(data, color='blue', label='Original')  # orig
        plt.plot(rolmean, color='red', label='Rolling Mean')  # mean
        plt.plot(rolstd, color='black', label='Rolling Std')  # std
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

        # Perform Dickey Fuller test
        result = adfuller(data)
        print('ADF Stastistic: %f' % result[0])
        print('p-value: %f' % result[1])
        # pvalue = result[1]
        for key, value in result[4].items():
            if result[0] > value:
                print("The graph is non stationary")
                break
            else:
                print("The graph is stationary")
                break
        print('Critical values:')
        for key, value in result[4].items():
            print('\t%s: %.3f ' % (key, value))

    def log_transforming(self, ts_log_data):
        plt.plot(ts_log_data, color="green")
        plt.show()

        self.test_stationarity(ts_log_data)

    def remove_trend_seasonality(self, ts_log_diff_data):
        plt.plot(ts_log_diff_data)
        plt.show()

        ts_log_diff.dropna(inplace=True)
        self.test_stationarity(ts_log_diff_data)

    def arima(self, ts_log_data, stationary_data):
        model = ARIMA(ts_log_data, order=(2, 1, 0))
        results_ARIMA = model.fit(disp=-1)
        plt.plot(stationary_data)
        plt.plot(results_ARIMA.fittedvalues, color='red')
        plt.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues - stationary_data) ** 2))
        # plt.show()

        size = int(len(ts_log_data) - 100)
        # Divide into train and test
        train_arima, test_arima = ts_log_data[0:size], ts_log_data[size:len(ts_log_data)]
        history = [x for x in train_arima]
        predictions = list()
        originals = list()
        error_list = list()

        print('Printing Predicted vs Expected Values...')
        print('\n')
        # We go over each value in the test set and then apply ARIMA model and calculate the predicted value.
        # We have the expected value in the test set, so we calculate the error between predicted and expected value
        sk_error_list = []
        for t in range(len(test_arima)):
            model = ARIMA(history, order=(2, 1, 0))
            model_fit = model.fit(disp=-1)

            output = model_fit.forecast()

            pred_value = output[0]

            original_value = test_arima[t]
            history.append(original_value)

            pred_value = dragon.exp(pred_value)

            original_value = dragon.exp(original_value)

            # Calculating the error
            error = ((abs(pred_value - original_value)) / original_value) * 100
            error_list.append(error)
            print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')

            predictions.append(float(pred_value))
            originals.append(float(original_value))

            sk_error = mean_absolute_error(originals, predictions)
            sk_error_list.append(sk_error)

        print((sum(sk_error_list) / len(sk_error_list)))

        # After iterating over whole test set the overall mean error is calculated.
        print('\n Mean Error in Predicting Test Case Articles : %f ' % (sum(error_list) / float(len(error_list))), '%')
        plt.figure(figsize=(8, 6))
        test_day = [t for t in range(len(test_arima))]
        labels = {'Orginal', 'Predicted'}
        plt.plot(test_day, predictions, color='green')
        plt.plot(test_day, originals, color='orange')
        plt.title('Expected Vs Predicted Views Forecasting')
        plt.xlabel('Day')
        plt.ylabel('Closing Price')
        plt.legend(labels)
        plt.show()


if __name__ == '__main__':
    close = FetchDataYahoo().training_set()['Close']
    # stationarity test
    ARIMA_Model().test_stationarity(close)
    # if not stationary
    ts_log = dragon.log(close)
    ARIMA_Model().log_transforming(ts_log)
    # if still not stationary: remove trend and seasonality with differencing
    ts_log_diff = ts_log - ts_log.shift()
    ARIMA_Model().remove_trend_seasonality(ts_log_diff)
    # the time serie is now stationary as p value < 0.05. Therefore we can now apply time series forecasting models.
    ARIMA_Model().arima(ts_log, ts_log_diff)


