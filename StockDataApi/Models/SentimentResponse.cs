namespace StockDataApi.Models
{
    public class SentimentResponse
    {
        public string Symbol { get; set; }
        public double SentimentScore { get; set; }
        public List<string> NewsArticles { get; set; }
    }
}
