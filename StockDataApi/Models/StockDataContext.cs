using Microsoft.EntityFrameworkCore;
using StockDataApi.Models;
namespace StockDataApi.Models

{
    public class StockDataContext : DbContext
    {

        public StockDataContext(DbContextOptions<StockDataContext> options) : base(options) { }
        public DbSet<Symbols> Symbols { get; set; }


        public DbSet<StockData> StockData { get; set; }
        public DbSet<StockSignal> StockSignals { get; set; }
        public DbSet<StockDataApi.Models.SentimentData> SentimentData { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
            {
                modelBuilder.Entity<StockData>()
                    .HasKey(sd => sd.Id);

                modelBuilder.Entity<StockSignal>()
                    .HasKey(ss => ss.Id);

                // Create a relationship between StockData and StockSignal on Symbol & Date
                modelBuilder.Entity<StockSignal>()
                    .HasOne<StockData>()
                    .WithMany()
                    .HasForeignKey(ss => new { ss.Symbol, ss.Date })
                    .HasPrincipalKey(sd => new { sd.Symbol, sd.Date });

                base.OnModelCreating(modelBuilder);
            }
    }
}
