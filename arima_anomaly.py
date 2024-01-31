import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt

headers = ['Time', 'PacketCounts']
dTypes = {'Time': 'str', 'PacketCounts': 'float'}
parseDates = ['Time']

networkData = []

def data_preprocessing():
    # Import all CSVs
    csvFiles = glob.glob("data/*.csv")
    dataList = (pd.read_csv(file, header=0) for file in csvFiles)
    networkData = pd.concat(dataList, ignore_index=True)

    # Converting DateFormat, Packets
    networkData['Time'] = pd.to_datetime(networkData['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    networkData.rename(columns={'Tot Fwd Pkts': 'PacketCounts'}, inplace=True)
    networkData['PacketCounts'] = networkData['PacketCounts'].astype(float)
    networkData = networkData.loc[:, ['Time','PacketCounts']]

    # Fill data indeces
    start = networkData.Time.min().replace(minute=0,second=0)
    end = networkData.Time.max()
    inex = pd.date_range(start, end, freq='60min')
    networkData.set_index('Time', inplace=True)
    networkData.index = pd.DatetimeIndex(networkData.index)
    networkData = networkData.resample('60min').mean()
    networkData = networkData.fillna(0)
    #networkData = networkData.reindex(inex, fill_value=0)

    # Export one CSV
    networkData.to_csv('norm_data.csv', columns = ['PacketCounts'])

def data_load():
    global networkData, trainData, testData, model
    networkData = pd.read_csv('norm_data.csv', index_col='Time', parse_dates=True)
    networkData.index.freq = "60min"

def data_plot():
    global networkData
    plt.plot(networkData.index, networkData['PacketCounts'], label='Actual')
    plt.title('Mean Network Traffic')
    plt.xlabel('Time')
    plt.ylabel('Packet Counts')
    plt.legend()
    plt.show()

def data_adf():
    global networkData
    dftest = adfuller(networkData, autolag = 'AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

def auto_ARIMA():
    global networkData
    stepwise_fit = auto_arima(networkData['PacketCounts'],
                              trace=True,
                              suppress_warnings=True)

def apply_ARIMA():
    global networkData
    order = (2, 0, 1)
    model = ARIMA(networkData['PacketCounts'], order=order)
    results = model.fit()

    start = networkData.index.min().replace(second=0)
    end = networkData.index.max()

    networkData['Predictions'] = results.predict(start=3, end=end, dynamic=False)
    
    # Plot
    plt.plot(networkData.index, networkData['PacketCounts'], label='Actual')
    plt.plot(networkData.index, networkData['Predictions'], label='Predicted')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Time')
    plt.ylabel('Packet Counts')
    plt.legend()
    plt.show()

def data_rmse():
    global networkData
    networkData = networkData.dropna()
    mean = networkData['PacketCounts'].mean()
    rmse = sqrt(mean_squared_error(networkData['Predictions'],networkData['PacketCounts']))
    vari = rmse/mean*100
    print("Dataset mean value: ", mean)
    print("Mean square error: ", rmse)
    print("Dataset variance: ", vari)
 
def data_anomaly_detection():
    global networkData
    
    # Fit ARIMA model
    order = (2, 0, 1)
    model = ARIMA(networkData['PacketCounts'], order=order)
    results = model.fit()

    start = networkData.index.min().replace(second=0)
    end = networkData.index.max()

    # Prediction
    networkData['Predictions'] = results.predict(start=3, end=end, dynamic=False)

    # Deviations
    networkData['Deviations'] = networkData['PacketCounts'] - networkData['Predictions']

    # Threshold for Anomaly Detection
    threshold = 3 * np.std(networkData['Deviations'])

    # Manual Anomalies
    networkData.loc['2022-12-01 12:00':'2022-12-01 14:00', 'PacketCounts'] *= 1.05
    networkData.loc['2022-12-03 08:00':'2022-12-03 17:00', 'PacketCounts'] *= 0.95
    networkData.loc['2022-12-04 10:00':'2022-12-04 15:00', 'PacketCounts'] *= 1.2
    networkData.loc['2022-12-07 07:00':'2022-12-07 11:00', 'PacketCounts'] *= 1.07

    # Recalculate Deviations
    networkData['Deviations'] = networkData['PacketCounts'] - networkData['Predictions']

    # Identify Anomalies
    networkData['Anomalies'] = np.abs(networkData['Deviations']) > threshold

    # Plot
    plt.plot(networkData.index, networkData['PacketCounts'], label='Actual')
    plt.plot(networkData.index, networkData['Predictions'], label='Predicted')
    plt.scatter(networkData.index[networkData['Anomalies']],
                networkData['PacketCounts'][networkData['Anomalies']], color='red', label='Anomaly')
    plt.title('Network Anomaly Detection using ARIMA')
    plt.xlabel('Time')
    plt.ylabel('Packet Counts')
    plt.legend()
    plt.show()

#data_preprocessing()
data_load()
#data_plot()
#data_adf()
#auto_ARIMA()
#apply_ARIMA()
#data_rmse()
data_anomaly_detection()
