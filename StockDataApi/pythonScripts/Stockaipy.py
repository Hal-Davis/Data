import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pyodbc
import datetime as df

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from textblob import TextBlob

# SQL Server Connection
server = "DESKTOP-FAVA9BI"
# database = "StockAIV3"
database = "StockAIVTest"
driver = "{SQL Server}"
conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
# ✅ Fetch Symbols from [dbo].[Symbols] Table
def fetch_symbols_from_db():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Symbol FROM [Symbols]")  
        rows = cursor.fetchall()
        conn.close()

        # Convert query results to a list and trim whitespace from nchar(20)
        symbols_from_db = [row[0].strip() for row in rows]

        if symbols_from_db:
            print(f"✅ Loaded {len(symbols_from_db)} symbols from [dbo].[Symbols].")
            return symbols_from_db
    except Exception as e:
        print(f"⚠️ ERROR: Could not fetch symbols from database - {e}")

    return None  # Return None if database fetch fails


# Set random seed for reproducibility
torch.manual_seed(0)

# Polygon.io API Key
POLYGON_API_KEY = "R1puZUIXEPiBm6MfppJNQwS4JxGJnNxL"
# News API Key
NEWS_API_KEY = "c29f29ac369b4117832b7c01f392da9e"

# SQL Server Connection
server = "DESKTOP-FAVA9BI"
# database = "StockAIV3"
database = 'StockAIVTest'
driver = "{SQL Server}"
conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"

# Track API call limits for Polygon.io
polygon_last_call = 0

# # Stock symbols list
# symbols = [
#     'CAT', 'SNDL', 'ZOM', 'OCGN', 'NXTP', 'BBIG', 'BHAT', 'TANH', 'AADT', 'AEHL', 'AIM', 'AAPL', 'MSFT', 'GOOGL',
#     'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'DIS', 'NFLX', 'PYPL', 'INTC',
#     'CSCO', 'AMD', 'XOM', 'KO', 'PEP', 'BA', 'NKE', 'GS', 'IBM', 'ADBE', 'CRM', 'T', 'VZ', 'MRNA', 'PFE',
#     'ABBV', 'CVX', 'CAT', 'GE', 'FDX', 'UPS', 'LMT', "PST"
# ]
symbols = fetch_symbols_from_db()

# Sequence length
sequence_length = 30  # Define sequence length for LSTM input
# Train ratio
train_ratio = 0.8  # Ratio of data used for training

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# LSTM parameters
input_size = 5
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 365

# RSI Calculation
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch data from Polygon.io
def fetch_polygon_data(symbol, max_retries=3):
    global polygon_last_call
    time.sleep(3)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2025-03-279"
    params = {"apiKey": POLYGON_API_KEY}
    time.sleep(4)
    for attempt in range(max_retries):
        wait_time = max(15 - (time.time() - polygon_last_call), 0)
        time.sleep(wait_time)
        response = requests.get(url, params=params)
        polygon_last_call = time.time()

        if response.status_code != 200:
            continue

        data = response.json()
        if "results" not in data:
            continue

        df = pd.DataFrame(data["results"])
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
        df["t"] = pd.to_datetime(df["t"], unit='ms')
        df.set_index("t", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    return None

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, 3]  # Predicting 'Close' price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
def fetch_stock_news(symbol):
    time.sleep(5)
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching news for {symbol}: {response.status_code}")
        return []

    articles = response.json().get("articles", [])
    return [(article["title"], article["description"]) for article in articles]

def analyze_sentiment(news_articles):
    positive, negative, neutral = 0, 0, 0

    for title, description in news_articles:
        text = f"{title}. {description}" if description else title
        sentiment = TextBlob(text).sentiment.polarity

        if sentiment > 0.1:
            positive += 1
        elif sentiment < -0.1:
            negative += 1
        else:
            neutral += 1

    total = positive + negative + neutral
    if total == 0:
        return 0
    
    print(f"Normalized Score{positive} - {negative} / {total} * 100")
    return (positive - negative) / total * 100  # Normalized score
    
    
def save_to_sql(symbol, sentiment_score, news_articles):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    news_text = "; ".join([title for title, _ in news_articles])

    query = """
    INSERT INTO SentimentData (Symbol, SentimentScore, NewsArticles, DateRecorded)
    VALUES (?, ?, ?, GetDate())
    """
 
    cursor.execute(query, (symbol, sentiment_score, news_text))

    conn.commit()
    conn.close()

# Timer loop to run every hour
while True:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    for symbol in symbols:
        print(f"Starting training for {symbol}...")
        print(f"Fetching news for {symbol}...")
        news_articles = fetch_stock_news(symbol)

        if not news_articles:
            print(f"No news found for {symbol}. Skipping...")
            continue

        sentiment_score = analyze_sentiment(news_articles)
        save_to_sql(symbol, sentiment_score, news_articles)
        print(f"Saved sentiment data for {symbol}: {sentiment_score:.2f}")

        time.sleep(2)  # Avoid hitting API rate limits
        
        try:
            df = fetch_polygon_data(symbol)
        except Exception as e:
            df = None

        if df is None or df.empty:
            print(f"No data available for {symbol}, skipping.")
            continue

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=True)

        # Calculate RSI and SMAs
        df['RSI'] = calculate_rsi(df)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df[['RSI', 'SMA_10', 'SMA_50']] = df[['RSI', 'SMA_10', 'SMA_50']].fillna(0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

        X, y = create_sequences(scaled_data, sequence_length)
        if len(X) == 0:
            print(f"Not enough data to create sequences for {symbol}, skipping.")
            continue

        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, num_epochs + 1):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == num_epochs:
                print(f"Epoch [{epoch}/{num_epochs}] for {symbol}, Loss: {loss.item():.4f}")

        print(f"Training completed for {symbol}.")

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy().flatten()
            y_actual = y_test.numpy().flatten()

        y_pred = scaler.inverse_transform(
            np.column_stack((np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred), y_pred, np.zeros_like(y_pred)))
        )[:, 3]

        y_actual = scaler.inverse_transform(
            np.column_stack((np.zeros_like(y_actual), np.zeros_like(y_actual), np.zeros_like(y_actual), y_actual, np.zeros_like(y_actual)))
        )[:, 3]
        time.sleep(15)

        # Insert predictions, stock data, and signals
        for i in range(len(y_pred)):
            date = df.index[-len(y_pred) + i]
            actual_price = float(y_actual[i])
            predicted_price = float(y_pred[i])

            buy_signal = int(predicted_price > actual_price * 1.02)
            sell_signal = int(predicted_price < actual_price * 0.98)
         
            cursor.execute("""
            MERGE INTO Symbols AS target 
            USING (SELECT ? AS Symbol) AS source
            ON target.Symbol = source.Symbol
            WHEN NOT MATCHED THEN
                INSERT (Symbol) VALUES (source.Symbol);
            """, (symbol,))
            
            cursor.execute(
                "INSERT INTO StockPredictions (Symbol, [Date], ActualPrice, PredictedPrice) VALUES (?, ?, ?, ?)",
                (symbol, date, actual_price, predicted_price)
            )

            cursor.execute(
                            "INSERT INTO StockData (Symbol, [Date], OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                symbol, 
                                date, 
                                float(df.iloc[-len(y_pred) + i]["Open"]), 
                                float(df.iloc[-len(y_pred) + i]["High"]), 
                                float(df.iloc[-len(y_pred) + i]["Low"]), 
                                float(df.iloc[-len(y_pred) + i]["Close"]), 
                                int(df.iloc[-len(y_pred) + i]["Volume"]),  # Fix: Ensuring only Volume is cast to int
                                int(y_actual[i])  # Fix: Ensuring Prediction is an integer
                            )

                        )

            cursor.execute(
                "INSERT INTO StockSignals (Symbol, [Date], PredictedPrice, BuySignal, SellSignal, RSI, SMA_10, SMA_50) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, date, predicted_price, buy_signal, sell_signal, 
                 float(df.iloc[-len(y_pred) + i]["RSI"]), 
                 float(df.iloc[-len(y_pred) + i]["SMA_10"]), 
                 float(df.iloc[-len(y_pred) + i]["SMA_50"]))
            )
            print(date)
      
            cursor.execute(
                """
                MERGE INTO StockStatistics AS Target
                USING (SELECT ? AS Symbol, ? AS MAE, ? AS MSE, ? AS RMSE, ? AS R2) AS Source
                ON Target.Symbol = Source.Symbol
                WHEN MATCHED THEN 
                    UPDATE SET MAE = Source.MAE, MSE = Source.MSE, RMSE = Source.RMSE, R2 = Source.R2
                WHEN NOT MATCHED THEN
                    INSERT (Symbol, MAE, MSE, RMSE, R2) 
                    VALUES (Source.Symbol, Source.MAE, Source.MSE, Source.RMSE, Source.R2);
                """,
                (symbol, mean_absolute_error(y_actual, y_pred), mean_squared_error(y_actual, y_pred), 
                np.sqrt(mean_squared_error(y_actual, y_pred)), r2_score(y_actual, y_pred))
            )


    
    print("All training completed. Sleeping for 1 hour...")
    conn.commit()
    conn.close()
    exit()
