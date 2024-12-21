import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("EVA-USD_historical_data.csv")

def calculate_volatility(df):
    df['Daily Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Change'].rolling(window=20).std()

calculate_volatility(df)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Volatility'], label='20-Day Volatility', color='red')
plt.title('20-Day Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid()
plt.show()
