﻿using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using StockDataApi.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Dynamic;


namespace StockDataApi.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class StockDataController : ControllerBase
    {
        private readonly StockDataContext _context;
        private readonly ILogger<StockDataController> _logger;
         
        public StockDataController(StockDataContext context, ILogger<StockDataController> logger)
        {
            _context = context;
            _logger = logger;
        }

        // ✅ Get Latest Stock Data
        [HttpGet]
        public async Task<ActionResult<IEnumerable<StockData>>> GetStockData()
        {
            return await _context.StockData
                .OrderByDescending(s => s.Date)
                .Take(100)
                .AsNoTracking()
                .ToListAsync();
        }
        [HttpGet("stock-data/{symbol}")]
        public async Task<ActionResult<IEnumerable<StockData>>> GetStockDataBySymbol(string symbol)
        {
            var stockData = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(100) // Adjust as needed
                .AsNoTracking()
                .ToListAsync();

            if (!stockData.Any())
                return NotFound("No stock data found for the symbol.");

            return Ok(stockData);
        }
        // ✅ Get Moving Average (SMA)
        [HttpGet("moving-average/{symbol}/{days}")]
        public async Task<ActionResult<double>> GetMovingAverage(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days)
                .Select(s => s.ClosePrice ?? 0) // Handle null values
                .ToListAsync();

            if (!stockPrices.Any()) return NotFound("No data available");

            return Ok(stockPrices.Average());
        }

        // ✅ Get Exponential Moving Average (EMA)
        [HttpGet("ema/{symbol}/{days}")]
        public async Task<ActionResult<double>> GetEMA(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days)
                .Select(s => s.ClosePrice ?? 0)
                .ToListAsync();

            if (!stockPrices.Any()) return NotFound("No data available");

            return Ok(CalculateEMA(stockPrices, days));
        }

        private double CalculateEMA(List<double> prices, int period)
        {
            if (prices.Count < period) return 0;

            double multiplier = 2.0 / (period + 1);
            double ema = prices.First();

            foreach (var price in prices.Skip(1))
            {
                ema = (price - ema) * multiplier + ema;
            }

            return ema;
        }

        // ✅ Get RSI
        [HttpGet("rsi/{symbol}/{days}")]
        public async Task<ActionResult<double>> GetRSI(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days + 1)
                .Select(s => s.ClosePrice ?? 0)
                .ToListAsync();

            if (stockPrices.Count < days) return NotFound("Not enough data");

            return Ok(CalculateRSI(stockPrices));
        }

        private double CalculateRSI(List<double> prices)
        {
            double gain = 0, loss = 0;

            for (int i = 1; i < prices.Count; i++)
            {
                double change = prices[i] - prices[i - 1];
                if (change > 0) gain += change;
                else loss -= change;
            }

            double avgGain = gain / prices.Count;
            double avgLoss = loss / prices.Count;
            if (avgLoss == 0) return 100;

            double rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        // ✅ Get Stock Symbols
        [HttpGet("symbols")]
        public async Task<ActionResult<List<string>>> GetDistinctSymbols()
        {
            return await _context.Symbols
                .Select(s => (s.Symbol ?? "").Trim())
                .Distinct()
                .OrderBy(s => s)
                .AsNoTracking()
                .ToListAsync();
        }

        [HttpGet("decision/{symbol}")]
        public async Task<ActionResult<string>> GetBuySellDecision(string symbol)
        {
            // ✅ Fetch stock signals, ensuring no duplicate dates
            var latestStockSignals = await _context.StockSignals
                .Where(ss => ss.Symbol == symbol)
                .OrderByDescending(ss => ss.Date)
                .GroupBy(ss => ss.Date) // ✅ Group by Date to prevent duplicate keys
                .Select(g => g.First()) // ✅ Take the most recent entry per date
                .ToDictionaryAsync(ss => ss.Date);

            // ✅ Fetch stock data
            var stockData = await _context.StockData
                .Where(sd => sd.Symbol == symbol)
                .OrderByDescending(sd => sd.Date)
                .Take(50)
                .ToListAsync();

            if (!stockData.Any()) return NotFound("Not enough data");

            // ✅ Get the latest stock data entry
            var latestEntry = stockData.FirstOrDefault();
            if (latestEntry == null || !latestStockSignals.TryGetValue(Convert.ToDateTime(latestEntry.Date), out var signal))
            {
                _logger.LogWarning($"No valid data available for {symbol} on {latestEntry?.Date}");
                return NotFound("No valid data available");
            }

            // ✅ Compute buy/sell decision
            bool buySignal = signal.SMA_10 > signal.SMA_50 && signal.RSI < 30;
            bool sellSignal = signal.SMA_10 < signal.SMA_50 && signal.RSI > 70;
            //return Ok(new { decision = buySignal ? "BUY" : sellSignal ? "SELL" : "HOLD" });
            return Ok(new { decision = buySignal ? "BUY" : sellSignal ? "SELL" : "HOLD" });
        }

        [HttpGet("momentum/{symbol}")]
        public async Task<ActionResult<int>> GetMomentumScore(string symbol)
        {
            var latestStockSignals = await _context.StockSignals
                .Where(ss => ss.Symbol == symbol)
                .OrderByDescending(ss => ss.Date)
                .GroupBy(ss => ss.Date) // Prevent duplicate dates
                .Select(g => g.First()) // Take latest per date
                .ToListAsync();

            if (!latestStockSignals.Any()) return NotFound("No stock signals available.");

            var stockData = await _context.StockData
                .Where(sd => sd.Symbol == symbol)
                .OrderByDescending(sd => sd.Date)
                .Take(50)
                .ToListAsync();

            if (!stockData.Any()) return NotFound("No stock data available.");

            var latestEntry = stockData.FirstOrDefault();
            if (latestEntry == null) return NotFound("No valid stock data available.");

            var previousEntry = stockData.Skip(1).FirstOrDefault(); // Get previous day's data
            if (previousEntry == null) return NotFound("Not enough data for momentum calculation.");

            var latestSignal = latestStockSignals.FirstOrDefault(s => s.Date == latestEntry.Date);
            var previousSignal = latestStockSignals.FirstOrDefault(s => s.Date == previousEntry.Date);

            if (latestSignal == null || previousSignal == null)
                return NotFound("No valid stock signal for the latest stock data.");

            // ✅ Compute momentum score with improved logic
            int score = 0;
            double closePrice = latestEntry.ClosePrice ?? 0.0;
            double prevClosePrice = previousEntry.ClosePrice ?? 0.0;
            if (latestSignal.SMA_50 > previousSignal.SMA_50) score += 1;
            bool bullishCrossover = previousSignal.SMA_10 < previousSignal.SMA_50 && latestSignal.SMA_10 > latestSignal.SMA_50;
            if (bullishCrossover) score += 1;
            if (latestSignal.RSI > previousSignal.RSI && latestSignal.RSI >= 50 && latestSignal.RSI < 70) score += 1;
            if (latestSignal.RSI < 30 && latestSignal.RSI < previousSignal.RSI) score -= 1;

            return Ok(score);
        }

        [HttpGet("atr/{symbol}/{days}")]
        public async Task<ActionResult<double>> GetATR(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days + 1)
                .Select(s => new { s.HighPrice, s.LowPrice, s.ClosePrice })
                .ToListAsync();

            if (stockPrices.Count < days) return NotFound("Not enough data");
            double atr = CalculateATR(stockPrices);
            return Ok(atr);
        }

        private double CalculateATR(dynamic prices)
        {
            double totalTR = 0;

            for (int i = 1; i < prices.Count; i++)
            {
                double highLow = prices[i].HighPrice - prices[i].LowPrice;
                double highPrevClose = Math.Abs(prices[i].HighPrice - prices[i - 1].ClosePrice);
                double lowPrevClose = Math.Abs(prices[i].LowPrice - prices[i - 1].ClosePrice);

                double trueRange = Math.Max(highLow, Math.Max(highPrevClose, lowPrevClose));
                totalTR += trueRange;
            }

            return totalTR / prices.Count;
        }
        [HttpGet("signal-accuracy/{symbol}/{days}")]
        public async Task<ActionResult<double>> GetSignalAccuracy(string symbol, int days)
        {
            var stockSignals = await _context.StockSignals
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days)
                .Select(s => new { s.BuySignal, s.SellSignal, s.PredictedPrice })
                .ToListAsync();

            var stockData = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days)
                .Select(s => s.ClosePrice)
                .ToListAsync();

            if (stockSignals.Count < days || stockData.Count < days)
                return NotFound("Not enough data");

            int correctSignals = 0;
            for (int i = 0; i < stockSignals.Count - 1; i++)
            {
                if (stockSignals[i].BuySignal && stockData[i + 1] > stockData[i])
                    correctSignals++;
                if (stockSignals[i].SellSignal && stockData[i + 1] < stockData[i])
                    correctSignals++;
            }

            double accuracy = (double)correctSignals / (stockSignals.Count) * 100;
            return Ok(accuracy);
        }

        [HttpGet("bollinger/{symbol}/{days}")]
        public async Task<ActionResult<object>> GetBollingerBands(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(days)
                .Select(s => s.ClosePrice)
                .ToListAsync();

            if (stockPrices.Count < days)
                return NotFound("Not enough data");

            double sma = stockPrices.Average() ?? 0;
            double stdDev = Math.Sqrt(stockPrices.Sum(p => Math.Pow((p ?? 0) - sma, 2)) / stockPrices.Count);

            double upperBand = sma + (2 * stdDev);
            double lowerBand = sma - (2 * stdDev);

            return Ok(new { UpperBand = upperBand, LowerBand = lowerBand, SMA = sma });
        }

        [HttpGet("bollinger-array/{symbol}/{days}")]
        public async Task<ActionResult<IEnumerable<object>>> GetBollingerBandsArray(string symbol, int days)
        {
            // Fetch stock data for the given symbol and order by date
            var stockData = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderBy(s => s.Date) // Sort in ascending order for correct calculations
                .ToListAsync();

            if (stockData.Count < days)
            {
                return NotFound("Not enough data to calculate Bollinger Bands.");
            }

            // List to hold the results
            var bollingerBandsArray = new List<object>();

            // Loop through the stock data with a sliding window of `days`
            for (int i = 0; i <= stockData.Count - days; i++)
            {
                // Take the relevant subset of stock prices for the sliding window
                var subset = stockData.Skip(i).Take(days).ToList();

                // Calculate SMA (Simple Moving Average)
                double sma = subset.Average(s => s.ClosePrice ?? 0);

                // Calculate standard deviation
                double stdDev = Math.Sqrt(subset.Sum(s => Math.Pow((s.ClosePrice ?? 0) - sma, 2)) / days);

                // Calculate upper and lower bands
                double upperBand = sma + (2 * stdDev);
                double lowerBand = sma - (2 * stdDev);

                // Add the Bollinger Bands for the last day in the window
                bollingerBandsArray.Add(new
                {
                    Date = subset.Last().Date,
                    UpperBand = upperBand,
                    LowerBand = lowerBand,
                    SMA = sma
                });
            }

            return Ok(bollingerBandsArray);
        }

        [HttpGet("macd/{symbol}")]
        public async Task<ActionResult<object>> GetMACD(string symbol)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(26)
                .Select(s => s.ClosePrice)
                .ToListAsync();

            if (stockPrices.Count < 26) return NotFound("Not enough data");

            double ema12 = CalculateEMA(stockPrices, 12);
            double ema26 = CalculateEMA(stockPrices, 26);
            double macd = ema12 - ema26;
            double signalLine = CalculateEMA(stockPrices.Take(9).ToList(), 9);

            return Ok(new { MACD = macd, SignalLine = signalLine });
        }

        private double CalculateEMA(List<double?> prices, int period)
        {
            if (!prices.Any() || prices.Count < period) return 0;

            double multiplier = 2.0 / (period + 1);
            double ema = prices.First() ?? 0;

            foreach (var price in prices.Skip(1))
            {
                ema = ((price ?? 0) - ema) * multiplier + ema;
            }

            return ema;
        }
        [HttpGet("sharpe/{symbol}")]
        public async Task<ActionResult<double>> GetSharpeRatio(string symbol)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(30)
                .Select(s => s.ClosePrice)
                .ToListAsync();

            if (stockPrices.Count < 30) return NotFound("Not enough data");

            double avgReturn = stockPrices.Average() ?? 0;
            double stdDev = Math.Sqrt(stockPrices.Sum(p => Math.Pow((p ?? 0) - avgReturn, 2)) / stockPrices.Count);

            const double riskFreeRate = 0.02; // Example: 2% annualized
            double sharpeRatio = (avgReturn - riskFreeRate) / stdDev;

            return Ok(sharpeRatio);
        }
        [HttpGet("sentiment/{symbol}")]
        public async Task<ActionResult<object>> GetSentimentForStock(string symbol)
        {
            var sentiment = await _context.SentimentData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.DateRecorded)
                .FirstOrDefaultAsync();

            if (sentiment == null)
                return NotFound("No sentiment data found for this stock.");

            return Ok(new
            {
                symbol = sentiment.Symbol,
                sentimentScore = sentiment.SentimentScore,
                newsArticles = sentiment.NewsArticles
                    .Split(new string[] { "; " }, StringSplitOptions.RemoveEmptyEntries)
            });
        }
        [HttpGet("ema-array/{symbol}/{days}")]
        public async Task<ActionResult<IEnumerable<double>>> GetEMAArray(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Select(s => s.ClosePrice ?? 0)
                .ToListAsync();

            if (stockPrices.Count < days) return NotFound("Not enough data");

            var emaArray = new List<double>();
            double multiplier = 2.0 / (days + 1);
            double ema = stockPrices.First();

            emaArray.Add(ema); // Seed with initial EMA
            foreach (var price in stockPrices.Skip(1))
            {
                ema = (price - ema) * multiplier + ema;
                emaArray.Add(ema);
            }

            return Ok(emaArray);
        }

    
        [HttpGet("sma-array/{symbol}/{days}")]
        public async Task<ActionResult<IEnumerable<double>>> GetSMAArray(string symbol, int days)
        {
            var stockPrices = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Select(s => s.ClosePrice ?? 0) // Handle null values
                .ToListAsync();

            if (stockPrices.Count < days) return NotFound("Not enough data");

            var smaArray = new List<double>();
            for (int i = 0; i <= stockPrices.Count - days; i++)
            {
                var window = stockPrices.Skip(i).Take(days);
                smaArray.Add(window.Take(i).Average());
            }

            return Ok(smaArray);
        }

        [HttpGet("fibonacci/{symbol}/{high}/{low}")]
        public IActionResult CalculateFibonacciLevels(string symbol, double high, double low)
        {
                        var levels = new List<object>
            {
            new { Level = "0%", Value = high },
            new { Level = "23.6%", Value = high - 0.236 * (high - low) },
            new { Level = "38.2%", Value = high - 0.382 * (high - low) },
            new { Level = "50%", Value = (high + low) / 2 },
            new { Level = "61.8%", Value = low + 0.618 * (high - low) },
            new { Level = "100%", Value = low }
            };
            return Ok(levels);
        }
       
        // ✅ Calculate Beta
        [HttpGet("beta/{symbol}")]
        public async Task<ActionResult<double>> CalculateBeta(string symbol)
        {
            var stockReturns = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Select(s => s.Prediction - (s.ClosePrice ?? 0)) // Example logic
                .ToListAsync();

            var marketReturns = await _context.StockStatistics
                .OrderByDescending(s => s.Timestamp)
                .Select(s => s.R2) // Assuming R2 represents market returns
                .ToListAsync();

            if (stockReturns.Count < 2 || marketReturns.Count < 2)
                return NotFound("Not enough data to calculate Beta.");

            double avgStockReturn = Convert.ToDouble(stockReturns.Average());
            double avgMarketReturn = marketReturns.Average();

            double covariance = stockReturns.Zip(marketReturns, (stock, market) => (stock - avgStockReturn) * (market - avgMarketReturn))
                                            .Sum() ?? 0 / (stockReturns.Count - 1);

            double marketVariance = marketReturns.Sum(market => Math.Pow(market - avgMarketReturn, 2)) / (marketReturns.Count - 1);

            double beta = covariance / marketVariance;
            return Ok(beta);
        }
        [HttpGet("stock-data-indicators/{symbol}/{days}")]
        public async Task<ActionResult<object>> GetStockDataWithIndicators(string symbol, int days)
        {
            var stockData = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderBy(s => s.Date)
                .Select(s => new { s.Date, ClosePrice = s.ClosePrice ?? 0 })
                .ToListAsync();

            if (stockData.Count < days)
                return NotFound("Not enough data");

            //var smaArray = new List<double>();
            //for (int i = 0; i <= stockData.Count - days; i++)
            //{
            //    var window = stockData.Skip(i).Take(days);
            //    smaArray.Add(window.Average(s => s.ClosePrice));
            //}
            // Calculate SMA
            var smaArray = new List<double?>();
            for (int i = 0; i < stockData.Count; i++)
            {
                if (i >= days - 1)
                {
                    var window = stockData.Skip(i - days + 1).Take(days);
                    smaArray.Add(window.Average(s => s.ClosePrice));
                }
                else
                {
                    smaArray.Add(null);
                }
            }

            // Calculate EMA
            var emaArray = new List<double>();
            double multiplier = 2.0 / (days + 1);
            double ema = stockData.First().ClosePrice;
            emaArray.Add(ema); // Seed with initial EMA
            foreach (var price in stockData.Skip(1).Select(s => s.ClosePrice))
            {
                ema = (price - ema) * multiplier + ema;
                emaArray.Add(ema);
            }

            // Prepare the result
            var result = stockData.Select((s, index) => new
            {
                s.Date,
                SMA = smaArray[index],
                EMA = emaArray[index]
            }).ToList();

            return Ok(result);
        }

        [HttpGet("vix/{symbol}")]
        public async Task<ActionResult<double>> GetVolatilityIndex(string symbol)
        {
            var stockData = await _context.StockData
                .Where(s => s.Symbol == symbol)
                .OrderByDescending(s => s.Date)
                .Take(30)
                .Select(s => s.ClosePrice ?? 0)
                .ToListAsync();

            if (!stockData.Any()) return NotFound("No data available");

            double vix = Math.Sqrt(stockData.Sum(price => Math.Pow(price - stockData.Average(), 2)) / stockData.Count);
            return Ok(vix);
        }
    }
}