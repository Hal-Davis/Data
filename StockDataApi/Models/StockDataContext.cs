using Microsoft.EntityFrameworkCore;
using StockDataApi.Models;
namespace StockDataApi.Models
    {
        public class StockDataContext : DbContext
        {
            public StockDataContext(DbContextOptions<StockDataContext> options) : base(options) { }

            // Existing tables
            public DbSet<Symbols> Symbols { get; set; }
            public DbSet<FundamentalData> FundamentalData { get; set; }
            public DbSet<StockData> StockData { get; set; }
            public DbSet<StockSignal> StockSignals { get; set; }
            public DbSet<SentimentData> SentimentData { get; set; }

            // New tables
            public DbSet<FibonacciLevel> FibonacciLevels { get; set; }
            public DbSet<BetaCalculation> BetaCalculations { get; set; }
            public DbSet<VolatilityIndex> VolatilityIndices { get; set; }
            public DbSet<StockStatistic> StockStatistics { get; set; } // Add StockStatistics
        protected override void OnModelCreating(ModelBuilder modelBuilder)
            {
                // Existing relationships
                modelBuilder.Entity<StockData>()
                    .HasKey(sd => sd.Id);

                modelBuilder.Entity<StockSignal>()
                    .HasKey(ss => ss.Id);

                modelBuilder.Entity<StockSignal>()
                    .HasOne<StockData>()
                    .WithMany()
                    .HasForeignKey(ss => new { ss.Symbol, ss.Date })
                    .HasPrincipalKey(sd => new { sd.Symbol, sd.Date });

                // New relationships
                modelBuilder.Entity<FibonacciLevel>()
                    .HasKey(fl => fl.Id);

                modelBuilder.Entity<BetaCalculation>()
                    .HasKey(bc => bc.Id);

                modelBuilder.Entity<VolatilityIndex>()
                    .HasKey(vi => vi.Id);
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

                // New: Configuration for StockStatistics
                modelBuilder.Entity<StockStatistic>()
                    .HasKey(ss => ss.Id); // Primary key for StockStatistic

            base.OnModelCreating(modelBuilder);


            }
        }
    }
//    public class StockDataContext : DbContext
//    {

//        public StockDataContext(DbContextOptions<StockDataContext> options) : base(options) { }
//        public DbSet<Symbols> Symbols { get; set; }

//        public DbSet<FundamentalData> FundamentalData { get; set; }

//        public DbSet<StockData> StockData { get; set; }
//        public DbSet<StockSignal> StockSignals { get; set; }
//        public DbSet<SentimentData> SentimentData { get; set; }
//        //PERatio
//        protected override void OnModelCreating(ModelBuilder modelBuilder)
//        {
//            modelBuilder.Entity<StockData>()
//                .HasKey(sd => sd.Id);

//            modelBuilder.Entity<StockSignal>()
//                .HasKey(ss => ss.Id);

//            // Create a relationship between StockData and StockSignal on Symbol & Date
//            modelBuilder.Entity<StockSignal>()
//                .HasOne<StockData>()
//                .WithMany()
//                .HasForeignKey(ss => new { ss.Symbol, ss.Date })
//                .HasPrincipalKey(sd => new { sd.Symbol, sd.Date });

//            base.OnModelCreating(modelBuilder);
//        }
////    }
//}
