using System;

namespace StockDataApi.Models
{
    public class FundamentalData
    {
        public int Id { get; set; } // Primary key
        public string Symbol { get; set; } // Stock symbol
        public DateTime Date { get; set; } // Date of financial data
        public decimal? EarningsPerShare { get; set; } // EPS value (nullable)
        public decimal? DividendsPerShare { get; set; } // Dividend payout per share (nullable)
        public decimal? TotalAssets { get; set; } // Total assets (nullable)
        public decimal? TotalLiabilities { get; set; } // Total liabilities (nullable)
        public int? SharesOutstanding { get; set; } // Number of outstanding shares (nullable)
    }
}