using Microsoft.EntityFrameworkCore;
using StockDataApi.Models;

var builder = WebApplication.CreateBuilder(args);

// ✅ Ensure the correct connection string key is used
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
if (string.IsNullOrEmpty(connectionString))
{
    throw new InvalidOperationException("Database connection string is missing!");
}

// ✅ Register the database
builder.Services.AddDbContext<StockDataContext>(options =>
    options.UseSqlServer(connectionString));

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// ✅ Define a single CORS policy that includes ALL allowed origins
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend",
        policy => policy.WithOrigins(
                        "http://localhost:4200", // ✅ Angular Frontend
                        "https://localhost:7163", // ✅ Blazor Frontend
                        "http://localhost:5002"  // ✅ Your API Calls
                    )
                    .AllowAnyHeader()
                    .AllowAnyMethod()
                    .AllowCredentials()); // Only needed if you're using authentication
});

var app = builder.Build();

// ✅ Apply CORS globally
app.UseCors("AllowFrontend");

// ✅ Enable Swagger in development mode
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

app.Run();
