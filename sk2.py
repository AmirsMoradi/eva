import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import timedelta

df = pd.read_csv("EVA-USD_historical_data.csv")


def add_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price_Lag_1'] = df['Close'].shift(1)
    df['Price_Lag_2'] = df['Close'].shift(2)
    df['Price_Lag_3'] = df['Close'].shift(3)

    if 'Volume' in df.columns:
        df['Volume_Lag_1'] = df['Volume'].shift(1)

    df = df.dropna()

    return df


def predict_price(df, future_days=7):
    df = add_features(df)
    df.loc[:, 'Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())

    X = df[['Date_ordinal', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3']].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    df.loc[:, 'Prediction'] = model.predict(
        scaler.transform(df[['Date_ordinal', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3']].values))

    last_date = df['Date'].iloc[-1]
    last_ordinal = last_date.toordinal()

    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    future_ordinals = [last_ordinal + i for i in range(1, future_days + 1)]

    last_price = df['Close'].iloc[-1]
    future_predictions = []

    for i in range(future_days):
        future_lag_1 = last_price
        future_lag_2 = df['Close'].iloc[-2] if len(df) > 1 else last_price
        future_lag_3 = df['Close'].iloc[-3] if len(df) > 2 else last_price

        X_future = np.array([[future_ordinals[i], future_lag_1, future_lag_2, future_lag_3]])
        X_future_scaled = scaler.transform(X_future)
        predicted_price = model.predict(X_future_scaled)[0]

        future_predictions.append(predicted_price)
        last_price = predicted_price

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions
    })

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Actual Price', color='blue')
    plt.plot(df['Date'], df['Prediction'], label='Predicted Price', color='green')
    plt.plot(future_df['Date'], future_df['Predicted_Price'], label='Future Predictions', color='red', linestyle='--')
    plt.title('Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


predict_price(df)
