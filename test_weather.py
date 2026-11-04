
from modules.weather import (
    get_weather_report,
    HARYANA_DISTRICTS
)

district = "Charkhi Dadri"

print("Available districts:", HARYANA_DISTRICTS)

report = get_weather_report(district)

print("\n===== KHETMITRA WEATHER REPORT =====")

print("\nDistrict:")
print(report["district"])

print("\nCoordinates:")
print(report["coordinates"])

print("\nCurrent weather:")
print(report["current_weather"])

print("\nSeven-day forecast:")
print(report["forecast"].to_string(index=False))

print("\nIrrigation advisory:")
print(report["irrigation"])
