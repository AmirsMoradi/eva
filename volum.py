import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv("EVA-USD_historical_data.csv")


def plot_price_vs_volume(df):

    plt.figure(figsize=(12, 6))
    plt.scatter(df['Volume'], df['Close'], alpha=0.5)
    plt.title('Price vs Volume')
    plt.xlabel('Volume')
    plt.ylabel('Close Price')
    plt.grid()
    plt.show()

plot_price_vs_volume(df)
