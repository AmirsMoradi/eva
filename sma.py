import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("EVA-USD_historical_data.csv")


def add_sma(df, window=20):

    df['SMA'] = df['Close'].rolling(window=window).mean()

add_sma(df, window=20)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.plot(df['Date'], df['SMA'], label='20-Day SMA', color='orange')
plt.title('Price with 20-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
