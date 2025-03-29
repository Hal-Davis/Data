namespace StockDataApi.Models
{
    public struct StockKey
    {
        public string Symbol { get; set; }
        public DateTime Date { get; set; }

        public override bool Equals(object? obj)
        {
            return obj is StockKey other && Symbol == other.Symbol && Date == other.Date;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Symbol, Date);
        }
    }
}
