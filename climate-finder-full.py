from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

def classify_koppen(monthly_temps, monthly_precips_cm, lat):
    """Calculates the Köppen-Geiger climate classification based on monthly normals.
    Expects monthly_temps in °C, monthly_precips_cm in cm, and latitude (to determine hemisphere).
    """
    mat = sum(monthly_temps) / 12.0
    map_precip = sum(monthly_precips_cm)
    tcold = min(monthly_temps)
    thot = max(monthly_temps)

    # Determine hemisphere and seasonal precipitation distribution
    is_northern = lat >= 0
    if is_northern:
        # Summer: Apr-Sept, Winter: Oct-Mar
        summer_precip = sum(monthly_precips_cm[3:9])
        winter_precip = sum(monthly_precips_cm[0:3]) + sum(monthly_precips_cm[9:12])
        s_months_precip = monthly_precips_cm[3:9]
        w_months_precip = monthly_precips_cm[0:3] + monthly_precips_cm[9:12]
    else:
        # Summer: Oct-Mar, Winter: Apr-Sept
        summer_precip = sum(monthly_precips_cm[0:3]) + sum(monthly_precips_cm[9:12])
        winter_precip = sum(monthly_precips_cm[3:9])
        s_months_precip = monthly_precips_cm[0:3] + monthly_precips_cm[9:12]
        w_months_precip = monthly_precips_cm[3:9]

    # Calculate the dry-season threshold (P_threshold in cm)
    if summer_precip >= 0.7 * map_precip:
        p_thresh = 2 * mat + 28
    elif winter_precip >= 0.7 * map_precip:
        p_thresh = 2 * mat
    else:
        p_thresh = 2 * mat + 14

    # Group B: Arid / Semi-Arid
    if map_precip < p_thresh:
        b_type = "BW" if map_precip < 0.5 * p_thresh else "BS"
        h_type = "h" if mat >= 18.0 else "k"
        desc = "Desert (Arid)" if b_type == "BW" else "Steppe (Semi-Arid)"
        zone_detail = "Hot" if h_type == "h" else "Cold"
        return f"{b_type}{h_type} ({zone_detail} {desc})"

    # Group E: Polar / Alpine
    if thot < 10.0:
        return "ET (Tundra)" if thot > 0.0 else "EF (Frost)"

    # Group A: Tropical
    if tcold >= 18.0:
        pdry = min(monthly_precips_cm)
        if pdry >= 6.0:
            return "Af (Tropical Rainforest)"
        elif map_precip >= 2.5 * (100 - pdry * 10) / 10:
            return "Am (Tropical Monsoon)"
        else:
            pdry_is_summer = pdry in s_months_precip
            sub_code = "As" if pdry_is_summer else "Aw"
            desc_detail = "Tropical Wet/Dry (Summer Dry)" if pdry_is_summer else "Tropical Wet/Dry (Winter Dry)"
            return f"{sub_code} ({desc_detail})"

    # Group C (Temperate) vs Group D (Continental)
    if 0.0 <= tcold < 18.0:
        group = "C"
    elif tcold < 0.0:
        group = "D"
    else:
        return "Unclassified"

    # Determine seasonal dry/wet parameters for C and D
    psdry = min(s_months_precip)
    pswet = max(s_months_precip)
    pwdry = min(w_months_precip)
    pwwet = max(w_months_precip)

    is_s = (psdry < 4.0) and (psdry < pwwet / 3.0)
    is_w = (pwdry < pswet / 10.0)

    if is_s and not is_w:
        season = "s"
    elif is_w and not is_s:
        season = "w"
    else:
        season = "f"

    # Determine temperature subtype classification
    thot_max = max(monthly_temps)
    warm_months_count = sum(1 for t in monthly_temps if t >= 10.0)

    if thot_max >= 22.0:
        temp_third = "a"
    elif warm_months_count >= 4:
        temp_third = "b"
    else:
        if group == "D" and tcold < -38.0:
            temp_third = "d"
        else:
            temp_third = "c"

    descriptions = {
        "Cfa": "Humid Subtropical",
        "Cfb": "Oceanic / Maritime Temperate",
        "Cfc": "Subpolar Oceanic",
        "Csa": "Hot-summer Mediterranean",
        "Csb": "Warm-summer Mediterranean",
        "Csc": "Cold-summer Mediterranean",
        "Cwa": "Monsoon-influenced Humid Subtropical",
        "Cwb": "Subtropical Highland",
        "Cwc": "Cold Subtropical Highland",
        "Dfa": "Hot-summer Humid Continental",
        "Dfb": "Warm-summer Humid Continental",
        "Dfc": "Subarctic / Boreal",
        "Dfd": "Extremely Cold Subarctic",
        "Dwa": "Monsoon-influenced Hot-summer Humid Continental",
        "Dwb": "Monsoon-influenced Warm-summer Humid Continental",
        "Dwc": "Monsoon-influenced Subarctic",
        "Dwd": "Monsoon-influenced Extremely Cold Subarctic",
        "Dsa": "Mediterranean-influenced Hot-summer Humid Continental",
        "Dsb": "Mediterranean-influenced Warm-summer Humid Continental",
        "Dsc": "Dry-summer Subarctic",
        "Dsd": "Extremely Cold Dry-summer Subarctic",
    }

    code = f"{group}{season}{temp_third}"
    desc = descriptions.get(code, "Temperate/Continental")
    return f"{code} ({desc})"


def get_climate_data(lat, lon, resolved_name):
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2014-01-01",
        "end_date": "2023-12-31",
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
    }

    try:
        arch_response = requests.get(archive_url, params=archive_params, timeout=10)

        if arch_response.status_code == 200:
            arch_data = arch_response.json().get("daily", {})
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(arch_data["time"]),
                    "temp": arch_data["temperature_2m_mean"],
                    "temp_max": arch_data["temperature_2m_max"], 
                    "temp_min": arch_data["temperature_2m_min"], 
                    "precip": arch_data["precipitation_sum"],
                }
            )

            df["month"] = df["date"].dt.month
            monthly_grouped = (
                df.groupby([df["date"].dt.year, "month"])
                .agg({"temp": "mean", "temp_max": "mean", "temp_min": "mean", "precip": "sum"})
                .reset_index()
            )

            monthly_normals = (
                monthly_grouped.groupby("month")
                .agg({"temp": "mean", "temp_max": "mean", "temp_min": "mean", "precip": "mean"})
                .reset_index()
            )

            t_list = monthly_normals["temp"].tolist()
            t_max_list = monthly_normals["temp_max"].tolist()
            t_min_list = monthly_normals["temp_min"].tolist()
            p_list_mm = monthly_normals["precip"].tolist()
            p_list_cm = [p / 10.0 for p in p_list_mm]

            koppen_result = classify_koppen(t_list, p_list_cm, lat)

            print(f"\nClimate normals retrieved successfully for: {resolved_name}")
            print(f"\n--- Köppen-Geiger Climate Classification ---")
            print(f"=> Climate Zone: {koppen_result}")
            print("\n- 10-Year Monthly Temperature Averages (°C) [Mean (Day Max / Night Min)] & Precipitation Totals (mm):")
            month_names = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
            for idx, m_name in enumerate(month_names):
                print(
                    f"  {m_name}: {t_list[idx]:.1f}°C (Day: {t_max_list[idx]:.1f}°C, Night: {t_min_list[idx]:.1f}°C), {p_list_mm[idx]:.1f} mm"
                )
        else:
            print(f"Failed to fetch archive data. Status code: {arch_response.status_code}")

    except Exception as e:
        print(f"Error processing climate data: {e}")


def resolve_city_to_coords(city_name, country=None):
    query = city_name.strip()
    if country and country.strip():
        query = f"{query}, {country.strip()}"

    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": query, "count": 1, "language": "en", "format": "json"}

    try:
        response = requests.get(geocoding_url, params=geo_params, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                loc = results[0]
                resolved_name = f"{loc.get('name', '')}, {loc.get('admin1', '')} ({loc.get('country', '')})"
                return loc["latitude"], loc["longitude"], resolved_name
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None, None


# --- User Interaction Menu ---
print("Select Input Mode:")
print("1. Search by City, Neighborhood, or Landmark")
print("2. Enter Coordinates Directly (Latitude, Longitude)")
mode_choice = input("Enter choice (1 or 2): ").strip()

if mode_choice == "2":
    coord_input = input(
        "Enter coordinates separated by a comma (e.g., 1.3521, 103.8198): "
    )
    try:
        lat_str, lon_str = coord_input.split(",")
        lat_val = float(lat_str.strip())
        lon_val = float(lon_str.strip())
        get_climate_data(
            lat_val, lon_val, f"Coordinates ({lat_val}, {lon_val})"
        )
    except ValueError:
        print("Invalid coordinate format. Please use numbers separated by a comma.")
else:
    city_input = input("Enter City, Neighborhood, or Landmark: ")
    country_input = input("Enter Country (optional, e.g. Canada, UK, US): ")

    if city_input.strip():
        lat, lon, resolved_name = resolve_city_to_coords(city_input, country_input)
        if lat is not None:
            get_climate_data(lat, lon, resolved_name)
        else:
            print(f"Could not find coordinates for location: '{city_input}'")
    else:
        print("Please provide at least a city or neighborhood name.")
