using System;
using System.ComponentModel.DataAnnotations;

namespace StockDataApi.Models
{
    public class StockData
    {
        [Key]
        public int Id { get; set; }
        public string? Symbol { get; set; }
        public DateTime? Date { get; set; }
        public double? OpenPrice { get; set; }
        public double? HighPrice { get; set; }
        public double? LowPrice { get; set; }
        public double? ClosePrice { get; set; }
        public long? Volume { get; set; }  // ✅ Change from double? to long?
        public DateTime? Timestamp { get; set; }
        public double? Prediction { get; set; }
    }
    
}

public class Symbols
{
    [Key]
    public string Symbol { get; set; }
}
public class StockSignal
    {
        public int Id { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public DateTime Date { get; set; }
        public double? PredictedPrice { get; set; }
        public bool BuySignal { get; set; }
        public bool SellSignal { get; set; }
        public double? RSI { get; set; }
        public double? SMA_10 { get; set; }
        public double? SMA_50 { get; set; }
    }
    
