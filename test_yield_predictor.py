
from modules.yield_predictor import (
    predict_wheat_yield,
    calculate_total_production,
    get_yield_model_info
)

# Inspect the saved model
print("\nMODEL INFORMATION")
print(get_yield_model_info())

# Example inputs
# Replace these with actual valid district-season data.
input_data = {
    "district": "Ambala",
    "year_start": 2004,
    "yield_lag_1": 3.26,
    "yield_lag_2": 3.56,
    "historical_avg_yield": 3.41,
    "rainfall_mm": 196.77,
    "temp_max_c": 25.6383,
    "temp_min_c": 12.8916,
    "humidity_pct": 46.1757
}

# Predict yield
result = predict_wheat_yield(input_data)

print("\nYIELD PREDICTION")
print(result)

# Calculate production for 5 acres
production = calculate_total_production(
    result["predicted_yield_tonnes_per_hectare"],
    5
)

print("\nFARM PRODUCTION")
print(production)
