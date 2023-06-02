import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pickle import load

st.title('Apple Sales Price Forecasting')


st.sidebar.write('Stock price prediction for upcoming N (N here can be any number of days) Days_')

st.sidebar.title("About :")
st.sidebar.subheader("Guided by: Neha Gupta")
st.sidebar.title("Team-4 :")
st.sidebar.subheader("UI by Sumit Musale") 

data_close = load(open('data_close.sav', 'rb'))
sarima_fit_final = load(open('model_trained.pkl', 'rb'))
periods = st.number_input('Number of Days',min_value=1)

datetime = pd.date_range('2020-01-01', periods=periods,freq='B')
date_df = pd.DataFrame(datetime,columns=['Date'])


forecast = sarima_fit_final.predict(len(data_close),len(data_close)+periods-1)
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['Stock Price']

data_forecast = forecast_df.set_index(date_df.Date)
st.success('Forecasting stock price value for '+str(periods)+' days')
st.write(data_forecast)



fig,ax = plt.subplots(figsize=(16,8),dpi=100)
ax.plot(data_close, label='Actual')
ax.plot(data_forecast,label='Forecast')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend(loc='upper left',fontsize=12)
ax.grid(True)
st.pyplot(fig)

