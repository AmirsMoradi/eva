import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("EVA-USD_historical_data.csv")

def plot_price_trend(df):

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.title('Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

plot_price_trend(df)
