import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("EVA-USD_historical_data.csv")


def predict_price(df):
    df['Date'] = pd.to_datetime(df['Date'])

    df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())

    X = df[['Date_ordinal']].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    df['Prediction'] = model.predict(X)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Actual Price', color='blue')
    plt.plot(df['Date'], df['Prediction'], label='Predicted Price', color='green')
    plt.title('Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


predict_price(df)
