using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace StockDataApi.Models
{
    [Table("SentimentData")]
    public class SentimentData
    {
        [Key]
        public int Id { get; set; }

        [Required]
        [StringLength(10)]
        public string Symbol { get; set; }

        [Required]
        public double SentimentScore { get; set; }

        [Required]
        public string NewsArticles { get; set; } // Stores concatenated news articles

        [Required]
        public DateTime DateRecorded { get; set; }
    }
}
