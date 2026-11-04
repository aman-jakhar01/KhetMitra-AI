
import requests
import pandas as pd

from functools import lru_cache


# ==========================================
# 1. CONFIGURATION
# ==========================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

TIMEZONE = "Asia/Kolkata"

REQUEST_TIMEOUT = 20


# ==========================================
# 2. HARYANA DISTRICTS
# ==========================================

HARYANA_DISTRICTS = [
    "Ambala",
    "Bhiwani",
    "Charkhi Dadri",
    "Faridabad",
    "Fatehabad",
    "Gurugram",
    "Hisar",
    "Jhajjar",
    "Jind",
    "Kaithal",
    "Karnal",
    "Kurukshetra",
    "Mahendragarh",
    "Nuh",
    "Palwal",
    "Panchkula",
    "Panipat",
    "Rewari",
    "Rohtak",
    "Sirsa",
    "Sonipat",
    "Yamunanagar"
]


# ==========================================
# 3. WEATHER CODE DESCRIPTION
# ==========================================

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail"
}


# ==========================================
# 4. GET DISTRICT COORDINATES
# ==========================================

@lru_cache(maxsize=64)
def get_district_coordinates(district):

    """
    Get latitude and longitude for a Haryana district.

    Coordinates are retrieved using Open-Meteo Geocoding API.
    """

    if district not in HARYANA_DISTRICTS:
        raise ValueError(
            f"Unsupported Haryana district: {district}"
        )

    params = {
        "name": district,
        "count": 10,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        GEOCODING_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        raise ValueError(
            f"Coordinates not found for {district}"
        )

    # Prefer a result in India.
    india_results = [
        item for item in results
        if item.get("country_code") == "IN"
    ]

    if not india_results:
        raise ValueError(
            f"Could not verify Indian location for {district}"
        )

    result = india_results[0]

    return {
        "district": district,
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"])
    }


# ==========================================
# 5. FETCH WEATHER DATA
# ==========================================

def get_weather(district):

    """
    Fetch current weather and 7-day forecast.
    """

    coordinates = get_district_coordinates(district)

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],

        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m"
        ],

        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration"
        ],

        "forecast_days": 7,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if "current" not in data or "daily" not in data:
        raise ValueError(
            "Weather API returned incomplete data."
        )

    return {
        "location": coordinates,
        "current": data["current"],
        "daily": data["daily"],
        "timezone": data.get("timezone", TIMEZONE)
    }


# ==========================================
# 6. PROCESS CURRENT WEATHER
# ==========================================

def process_current_weather(weather_data):

    current = weather_data["current"]

    weather_code = int(
        current.get("weather_code", -1)
    )

    return {
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "precipitation_mm": current.get("precipitation"),
        "condition": WEATHER_CODES.get(
            weather_code, "Unknown"
        ),
        "observation_time": current.get("time")
    }


# ==========================================
# 7. PROCESS 7-DAY FORECAST
# ==========================================

def process_forecast(weather_data):

    daily = weather_data["daily"]

    forecast_df = pd.DataFrame({
        "Date": daily["time"],

        "Condition": [
            WEATHER_CODES.get(int(code), "Unknown")
            for code in daily["weather_code"]
        ],

        "Max Temperature (°C)": daily["temperature_2m_max"],

        "Min Temperature (°C)": daily["temperature_2m_min"],

        "Rainfall (mm)": daily["precipitation_sum"],

        "Rain chance (%)": daily["precipitation_probability_max"],

        "Wind speed (km/h)": daily["wind_speed_10m_max"],

        "ET0 (mm)": daily["et0_fao_evapotranspiration"]
    })

    numeric_columns = [
        "Max Temperature (°C)",
        "Min Temperature (°C)",
        "Rainfall (mm)",
        "Rain chance (%)",
        "Wind speed (km/h)",
        "ET0 (mm)"
    ]

    for column in numeric_columns:
        forecast_df[column] = pd.to_numeric(
            forecast_df[column],
            errors="coerce"
        )

    return forecast_df


# ==========================================
# 8. CALCULATE WEATHER WATER BALANCE
# ==========================================

def calculate_water_balance(forecast_df):

    """
    Compare forecast rainfall with reference ET0.

    This is a weather indicator, NOT the exact crop
    irrigation requirement.
    """

    total_rainfall = float(
        forecast_df["Rainfall (mm)"].fillna(0).sum()
    )

    total_et0 = float(
        forecast_df["ET0 (mm)"].fillna(0).sum()
    )

    water_balance = total_rainfall - total_et0

    return {
        "total_rainfall_mm": round(total_rainfall, 2),
        "total_et0_mm": round(total_et0, 2),
        "weather_water_balance_mm": round(
            water_balance, 2
        )
    }


# ==========================================
# 9. IRRIGATION ADVISORY
# ==========================================

def get_irrigation_advisory(forecast_df):

    """
    Generate a simple weather-based irrigation advisory.

    Does not replace soil moisture measurements or
    crop-stage-based irrigation scheduling.
    """

    balance = calculate_water_balance(forecast_df)

    total_rainfall = balance["total_rainfall_mm"]

    total_et0 = balance["total_et0_mm"]

    rain_chance = forecast_df[
        "Rain chance (%)"
    ].max()

    if pd.isna(rain_chance):
        rain_chance = 0

    if total_rainfall >= total_et0 and total_rainfall > 0:

        advisory = (
            "Forecast rainfall is at least as high as "
            "reference ET0. Check field drainage and "
            "soil moisture before irrigating."
        )

        status = "Rainfall may offset atmospheric demand"

    elif total_rainfall < total_et0 and rain_chance < 30:

        advisory = (
            "Forecast rainfall is below reference ET0. "
            "Check soil moisture and wheat growth stage. "
            "Irrigation may be needed if the root zone is dry."
        )

        status = "Check irrigation requirement"

    else:

        advisory = (
            "Forecast rainfall may partly offset atmospheric "
            "water demand. Monitor soil moisture and review "
            "the forecast before scheduling irrigation."
        )

        status = "Monitor field conditions"

    return {
        "status": status,
        "advisory": advisory,
        **balance
    }


# ==========================================
# 10. COMPLETE WEATHER SERVICE
# ==========================================

def get_weather_report(district):

    """
    Main function used by the Streamlit application.
    """

    weather_data = get_weather(district)

    current_weather = process_current_weather(
        weather_data
    )

    forecast_df = process_forecast(
        weather_data
    )

    irrigation = get_irrigation_advisory(
        forecast_df
    )

    return {
        "district": district,
        "coordinates": weather_data["location"],
        "current_weather": current_weather,
        "forecast": forecast_df,
        "irrigation": irrigation,
        "timezone": weather_data["timezone"]
    }
