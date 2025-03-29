import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pyodbc
import feedparser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set random seed for reproducibility
torch.manual_seed(0)

# Polygon.io API Key
POLYGON_API_KEY = "R1puZUIXEPiBm6MfppJNQwS4JxGJnNxL"

# SQL Server Connection
server = "Your SQL Server Here"
database = "StockAIVTest"
driver = "{SQL Server}"
conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
# ✅ Fetch Symbols from [dbo].[Symbols] Table
def fetch_symbols_from_db():
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT Symbol FROM [StockAI].[dbo].[Symbols]")  
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

# ✅ Load Symbols Dynamically from [dbo].[Symbols]
symbols = fetch_symbols_from_db() or []
# Track API call limits for Polygon.io
polygon_last_call = 0

# Stock symbols list
# symbols = [
#     'SNDL', 'ZOM', 'OCGN', 'NXTP', 'BBIG', 'BHAT', 'TANH', 'AADT', 'AEHL', 'AIM', 'AAPL', 'MSFT', 'GOOGL',
#     'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B', 'JNJ', 'JPM', 'V', 'WMT', 'PG', 'DIS', 'NFLX', 'PYPL', 'INTC',
#     'CSCO', 'AMD', 'XOM', 'KO', 'PEP', 'BA', 'NKE', 'GS', 'IBM', 'ADBE', 'CRM', 'T', 'VZ', 'MRNA', 'PFE',
#     'ABBV', 'CVX', 'CAT', 'GE', 'FDX', 'UPS', 'LMT', "PST"
# ]


# Sequence length
sequence_length = 10  
input_size = 6  # Now includes Sentiment Score
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 100

# ✅ Load FinBERT for Sentiment Analysis
finbert_model = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(finbert_model)
model = AutoModelForSequenceClassification.from_pretrained(finbert_model)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework="pt")

# ✅ Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last LSTM output for prediction
        return out


# RSI Calculation
def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ✅ Fetch News Headlines & Sentiment Analysis
def fetch_sentiment(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = [(entry.title, entry.summary) for entry in feed.entries[:5]]

    sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
    
    for title, description in articles:
        try:
            text = f"{title}. {description}" if description else title
            result = sentiment_pipeline(text) 
        except Exception as e:
                print(f"⚠️ Warn: We were unable to process {symbol}: Error Message:{e}")
                continue
            
        for sentiment in result:
            label = sentiment["label"].lower()
            if label in sentiment_scores:
                sentiment_scores[label] += sentiment["score"]

    total = sum(sentiment_scores.values())
    return ((sentiment_scores["positive"] - sentiment_scores["negative"]) / total * 100) if total else 0, articles

# ✅ Fetch data from Polygon.io
def fetch_polygon_data(symbol, max_retries=3):
    global polygon_last_call
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2025-01-01/2025-03-25"
    params = {"apiKey": POLYGON_API_KEY}

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
        y = data[i + seq_length, 3]  
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Timer loop to run every hour
while True:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    for symbol in symbols:
        print(f"Starting training for {symbol}...")
        time.sleep(2)

        sentiment_score, news_articles = fetch_sentiment(symbol)
        news_text = "; ".join([title for title, _ in news_articles])

        df = fetch_polygon_data(symbol)

        # ✅ If Polygon.io data is unavailable, fallback to Yahoo Finance
        if df is None or df.empty:
            print(f"Polygon.io data unavailable. Fetching from Yahoo Finance for {symbol}...")
            df = yf.download(symbol, start='2000-01-01', end='2029-01-01', progress=False)

        if df is None or df.empty:
            continue

        df['Sentiment'] = sentiment_score  
        df['RSI'] = calculate_rsi(df)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df.fillna(0, inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']])

        X, y = create_sequences(scaled_data, sequence_length)
        if len(X) == 0:
            continue

        X_train = torch.tensor(X[:int(len(X) * 0.8)], dtype=torch.float32)
        y_train = torch.tensor(y[:int(len(y) * 0.8)], dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X[int(len(X) * 0.8):], dtype=torch.float32)
        y_test = torch.tensor(y[int(len(y) * 0.8):], dtype=torch.float32).view(-1, 1)

        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        for epoch in range(1, num_epochs + 1):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == num_epochs:
                print(f"Epoch [{epoch}/{num_epochs}] for {symbol}, Loss: {loss.item():.4f}")

        print(f"Training completed for {symbol}.")


       # ✅ Ensure y_actual is defined
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy().flatten()
            y_actual = y_test.numpy().flatten()  # ✅ Fix applied

        print(f"DEBUG: y_actual length = {len(y_actual)}, y_pred length = {len(y_pred)}")  # ✅ Debugging

        # ✅ Train Model
        for i in range(len(y_pred)):
            date = df.index[-len(y_pred) + i]
            actual_price = float(y_actual[i]) if not np.isnan(y_actual[i]) else 0.0
            predicted_price = float(y_pred[i])
        # ✅ Fix: Define buy and sell signals before inserting into SQL
            buy_signal = int(predicted_price > actual_price * 1.02)  
            sell_signal = int(predicted_price < actual_price * 0.98)  
            
                        
        cursor.execute("""
            MERGE INTO Symbols AS Target
            USING (SELECT ? AS Symbol) AS Source
            ON Target.Symbol = Source.Symbol
            WHEN MATCHED THEN
                UPDATE SET Symbol = trim(Source.Symbol)
            WHEN NOT MATCHED THEN
                INSERT (Symbol)
                VALUES (trim(Source.Symbol));
        """, (symbol))

        cursor.execute("""
            MERGE INTO StockPredictions AS Target
            USING (SELECT ? AS Symbol, ? AS [Date], ? AS ActualPrice, ? AS PredictedPrice, ? AS SentimentScore) AS Source
            ON Target.Symbol = Source.Symbol AND Target.[Date] = Source.[Date]
            WHEN MATCHED THEN
                UPDATE SET ActualPrice = Source.ActualPrice, PredictedPrice = Source.PredictedPrice, SentimentScore = Source.SentimentScore
            WHEN NOT MATCHED THEN
                INSERT (Symbol, [Date], ActualPrice, PredictedPrice, SentimentScore)
                VALUES (trim(Source.Symbol), Source.[Date], Source.ActualPrice, Source.PredictedPrice, Source.SentimentScore);
        """, (symbol, date, actual_price, predicted_price, sentiment_score))

        cursor.execute("""
            MERGE INTO SentimentData AS Target
            USING (SELECT ? AS Symbol, ? AS SentimentScore, ? AS NewsArticles) AS Source
            ON Target.Symbol = Source.Symbol
            WHEN MATCHED THEN
                UPDATE SET SentimentScore = Source.SentimentScore, NewsArticles = Source.NewsArticles
            WHEN NOT MATCHED THEN
                INSERT (Symbol, SentimentScore, NewsArticles)
                VALUES (trim(Source.Symbol), Source.SentimentScore, Source.NewsArticles);
        """, (symbol, sentiment_score, news_text))

        cursor.execute("""
            MERGE INTO StockData AS Target
            USING (SELECT ? AS Symbol, ? AS [Date], ? AS OpenPrice, ? AS HighPrice, ? AS LowPrice, ? AS ClosePrice, ? AS Volume, ? AS SentimentScore) AS Source
            ON Target.Symbol = Source.Symbol AND Target.[Date] = Source.[Date]
            WHEN MATCHED THEN
                UPDATE SET OpenPrice = Source.OpenPrice, HighPrice = Source.HighPrice, LowPrice = Source.LowPrice, ClosePrice = Source.ClosePrice, Volume = Source.Volume, SentimentScore = Source.SentimentScore
            WHEN NOT MATCHED THEN
                INSERT (Symbol, [Date], OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, SentimentScore)
                VALUES (trim(Source.Symbol), Source.[Date], Source.OpenPrice, Source.HighPrice, Source.LowPrice, Source.ClosePrice, Source.Volume, Source.SentimentScore);
        """, (symbol, date, df.iloc[-len(y_pred) + i]["Open"], df.iloc[-len(y_pred) + i]["High"],
            df.iloc[-len(y_pred) + i]["Low"], df.iloc[-len(y_pred) + i]["Close"],
            df.iloc[-len(y_pred) + i]["Volume"], sentiment_score))

        cursor.execute("""
            MERGE INTO StockStatistics AS Target
            USING (SELECT ? AS Symbol, ? AS MAE, ? AS MSE, ? AS RMSE, ? AS R2, ? AS SentimentScore) AS Source
            ON Target.Symbol = Source.Symbol
            WHEN MATCHED THEN
                UPDATE SET MAE = Source.MAE, MSE = Source.MSE, RMSE = Source.RMSE, R2 = Source.R2, SentimentScore = Source.SentimentScore
            WHEN NOT MATCHED THEN
                INSERT (Symbol, MAE, MSE, RMSE, R2, SentimentScore)
                VALUES (trim(Source.Symbol), Source.MAE, Source.MSE, Source.RMSE, Source.R2, Source.SentimentScore);
        """, (symbol, mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred),
            np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred), sentiment_score))

        cursor.execute("""
            MERGE INTO StockSignals AS Target
            USING (SELECT ? AS Symbol, ? AS [Date], ? AS PredictedPrice, ? AS BuySignal, ? AS SellSignal, ? AS RSI, ? AS SMA_10, ? AS SMA_50, ? AS SentimentScore) AS Source
            ON Target.Symbol = Source.Symbol AND Target.[Date] = Source.[Date]
            WHEN MATCHED THEN
                UPDATE SET PredictedPrice = Source.PredictedPrice, BuySignal = Source.BuySignal, SellSignal = Source.SellSignal, RSI = Source.RSI, SMA_10 = Source.SMA_10, SMA_50 = Source.SMA_50, SentimentScore = Source.SentimentScore
            WHEN NOT MATCHED THEN
                INSERT (Symbol, [Date], PredictedPrice, BuySignal, SellSignal, RSI, SMA_10, SMA_50, SentimentScore)
                VALUES (trim(Source.Symbol), Source.[Date], Source.PredictedPrice, Source.BuySignal, Source.SellSignal, Source.RSI, Source.SMA_10, Source.SMA_50, Source.SentimentScore);
        """, (symbol, date, predicted_price, buy_signal, sell_signal,
            float(df.iloc[-len(y_pred) + i]["RSI"]),
            float(df.iloc[-len(y_pred) + i]["SMA_10"]),
            float(df.iloc[-len(y_pred) + i]["SMA_50"]),
            sentiment_score))
        
       

    conn.commit()
    conn.close()
    print("All training completed. Sleeping for 1 hour...")
    exit()
