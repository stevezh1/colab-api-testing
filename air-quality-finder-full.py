from datetime import datetime, timedelta, timezone
import requests

# Test Air Quality API connection with placeholder coordinates (London, UK)
url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": 0.0000,
    "longitude": 0.0000,
    "current": ["us_aqi", "pm2_5", "pm10", "nitrogen_dioxide"],
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    current_aqi = data["current"]
    print(
        f"Testing Air Quality API connection: 0.0000, 0.0000, Atlantic Ocean"
    )
    print(f"US AQI: {current_aqi.get('us_aqi', 'N/A')}")
    print(f"PM2.5: {current_aqi.get('pm2_5', 'N/A')} µg/m³")
    print(f"PM10: {current_aqi.get('pm10', 'N/A')} µg/m³")
    print(f"API successfully connected!\n")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


def get_aqi_by_city(city_name):
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    geo_response = requests.get(geocoding_url, params=geo_params)
    if geo_response.status_code != 200:
        print(f"Geocoding API failed with status code {geo_response.status_code}")
        return None

    geo_data = geo_response.json()
    if "results" not in geo_data or len(geo_data["results"]) == 0:
        print(f"Could not find coordinates for city: '{city_name}'")
        return None

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = f"{location.get('name', '')}, {location.get('admin1', '')} ({location.get('country', '')})"

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "us_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
    }

    aqi_response = requests.get(aqi_url, params=aqi_params)
    if aqi_response.status_code != 200:
        print(f"Air Quality API failed with status code {aqi_response.status_code}")
        return None

    aqi_data = aqi_response.json()
    current = aqi_data["current"]

    return {
        "resolved_name": resolved_name,
        "us_aqi": current.get("us_aqi"),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "co": current.get("carbon_monoxide"),
        "no2": current.get("nitrogen_dioxide"),
        "so2": current.get("sulphur_dioxide"),
        "o3": current.get("ozone"),
    }


def get_aqi_risk_level(aqi):
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "🟢 Good (Air quality is satisfactory, little or no risk)"
    elif aqi <= 100:
        return (
            "🟡 Moderate (Acceptable; sensitive individuals may experience minor"
            " issues)"
        )
    elif aqi <= 150:
        return (
            "🟠 Unhealthy for Sensitive Groups (Members of sensitive groups"
            " may experience health effects)"
        )
    elif aqi <= 200:
        return (
            "🔴 Unhealthy (Everyone may begin to experience health effects)"
        )
    elif aqi <= 300:
        return (
            "🟣 Very Unhealthy (Health alert: everyone may experience more"
            " serious health effects)"
        )
    else:
        return (
            "❌ Hazardous (Health warnings of emergency conditions. Entire"
            " population affected)"
        )


class TimezoneRepository:

    @staticmethod
    def get_local_time_and_offset(timezone_name):
        if not timezone_name or timezone_name == "UTC":
            return datetime.now(timezone.utc).strftime("%H:%M:%S"), "GMT+00:00"

        if "Singapore" in timezone_name or "Singapore" in str(timezone_name):
            local_now = datetime.now(timezone.utc) + timedelta(hours=8)
            return local_now.strftime("%H:%M:%S"), "GMT+08:00"

        utc_now = datetime.now(timezone.utc)
        return utc_now.strftime("%H:%M:%S"), "GMT+00:00"


tz_repository = TimezoneRepository()


def get_global_aqi_disambiguated(city_or_neighborhood, country=None):
    query = city_or_neighborhood.strip()
    if country and country.strip():
        query = f"{query}, {country.strip()}"

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": query, "count": 1, "language": "en", "format": "json"}

    resolved_timezone = "UTC"
    utc_offset_seconds = 0

    try:
        geo_response = requests.get(geocoding_url, params=geo_params, timeout=5)
        if geo_response.status_code == 200:
            results = geo_response.json().get("results", [])
            if results:
                lat = results[0]["latitude"]
                lon = results[0]["longitude"]

                aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
                aqi_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": "auto",
                }
                aqi_response = requests.get(
                    aqi_url, params=aqi_params, timeout=5
                )
                if aqi_response.status_code == 200:
                    a_data = aqi_response.json()
                    resolved_timezone = a_data.get("timezone", "UTC")
                    utc_offset_seconds = a_data.get("utc_offset_seconds", 0)
    except Exception:
        pass

    hours = int(abs(utc_offset_seconds) // 3600)
    minutes = int((abs(utc_offset_seconds) % 3600) // 60)
    sign = "+" if utc_offset_seconds >= 0 else "-"
    gmt_offset = f"GMT{sign}{hours:02d}:{minutes:02d}"

    local_dt = datetime.now(timezone.utc) + timedelta(seconds=utc_offset_seconds)
    local_time = local_dt.strftime("%H:%M:%S")
    current_date = local_dt.strftime("%Y-%m-%d")

    timestamp = f"{current_date} {local_time} {gmt_offset}"

    print(f'[{timestamp}] Searching Air Quality for: "{query}"')

    analysis = get_aqi_by_city(query)
    if analysis:
        risk_evaluation = get_aqi_risk_level(analysis["us_aqi"])
        print(f"\nAir Quality Index (AQI) Analysis for {analysis['resolved_name']}:")
        print(f"- Timezone: {resolved_timezone}")
        print(f"- U.S. AQI: {analysis['us_aqi']}")
        print(f"- PM2.5: {analysis['pm2_5']} µg/m³")
        print(f"- PM10: {analysis['pm10']} µg/m³")
        print(f"- Nitrogen Dioxide (NO2): {analysis['no2']} µg/m³")
        print(f"- Sulphur Dioxide (SO2): {analysis['so2']} µg/m³")
        print(f"- Ozone (O3): {analysis['o3']} µg/m³")
        print(f"- Carbon Monoxide (CO): {analysis['co']} µg/m³")
        print(f"=> Health Risk Level: {risk_evaluation}")
    else:
        print("Error: Air quality data could not be parsed or location was not found.")


city_input = input("Enter City, Neighborhood, or Landmark: ")
country_input = input("Enter Country (optional, e.g. Canada, UK, US): ")

if city_input.strip():
    get_global_aqi_disambiguated(city_input, country_input)
else:
    print("Please provide at least a city or neighborhood name.")

