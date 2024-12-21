import yfinance as yf
import pandas as pd


def fetch_crypto_data(symbol, start_date, end_date):

    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

        if not data.empty:
            data.reset_index(inplace=True)
            print(f"Data fetched for {symbol}.")
            return data
        else:
            print(f"No data found for {symbol}.")
            return pd.DataFrame()
    except Exception as e:
        print("Error:", e)
        return pd.DataFrame()


symbol = "EVA-USD"
start_date = "2020-01-01"
end_date = "2024-12-31"

df = fetch_crypto_data(symbol, start_date, end_date)

if not df.empty:
    df.to_csv(f'{symbol}_historical_data.csv', index=False)
    print(f"Data saved to {symbol}_historical_data.csv")
else:
    print("No data was fetched.")
