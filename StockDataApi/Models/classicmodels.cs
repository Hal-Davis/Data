public class FibonacciLevel
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public decimal HighPrice { get; set; }
    public decimal LowPrice { get; set; }
    public required string RetracementLevel { get; set; }
    public decimal LevelValue { get; set; }
    public DateTime CalculatedOn { get; set; } = DateTime.UtcNow;
}
public class BetaCalculation
{
    public int Id { get; set; }
    public string? Symbol { get; set; }
    public decimal BetaValue { get; set; }
    public int Period { get; set; } // Number of data points used
    public DateTime CalculatedOn { get; set; } = DateTime.UtcNow;
}
public class VolatilityIndex
{
    public int Id { get; set; }
    public string? Symbol { get; set; }
    public decimal VIXValue { get; set; }
    public DateTime CalculatedOn { get; set; } = DateTime.UtcNow;
}
public class StockStatistic
{
    public int Id { get; set; }
    public string? Symbol { get; set; } // NVARCHAR(10)
    public double MAE { get; set; } // Mean Absolute Error
    public double MSE { get; set; } // Mean Squared Error
    public double RMSE { get; set; } // Root Mean Squared Error
    public double R2 { get; set; } // R-squared value
    public DateTime Timestamp { get; set; } // Date and time of calculation
    public float? SentimentScore { get; set; } // Nullable Sentiment Score
}