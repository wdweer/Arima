import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 데이터 불러오기
data = pd.read_csv('~/Downloads/simulated_timeseries_data.csv', parse_dates=['Date'], index_col='Date')

# 데이터의 안정성 확인 (ADF Test)
result = adfuller(data['Value'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# 데이터 차분 (필요한 경우)
data_diff = data.diff().dropna()
print(data['Value'])
# ARIMA 모델 피팅
model = ARIMA(data['Value'], order=(10, 2, 10))  # p, d, q 값 설정
model_fit = model.fit()

# 모델 요약
print(model_fit.summary())

# 예측 및 시각화
forecast = model_fit.forecast(steps=10)
data['Value'].plot(label='Original')
forecast.plot(label='Forecast', color='red')
plt.legend()
plt.show()