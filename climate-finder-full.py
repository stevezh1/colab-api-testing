from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests

def classify_koppen(monthly_temps, monthly_precips_cm, lat):
    """Calculates the Köppen-Geiger climate classification based on monthly normals."""
    mat = sum(monthly_temps) / 12.0
    map_precip = sum(monthly_precips_cm)
    tcold = min(monthly_temps)
    thot = max(monthly_temps)

    is_northern = lat >= 0
    if is_northern:
        summer_precip = sum(monthly_precips_cm[3:9])
        winter_precip = sum(monthly_precips_cm[0:3]) + sum(monthly_precips_cm[9:12])
        s_months_precip = monthly_precips_cm[3:9]
        w_months_precip = monthly_precips_cm[0:3] + monthly_precips_cm[9:12]
    else:
        summer_precip = sum(monthly_precips_cm[0:3]) + sum(monthly_precips_cm[9:12])
        winter_precip = sum(monthly_precips_cm[3:9])
        s_months_precip = monthly_precips_cm[0:3] + monthly_precips_cm[9:12]
        w_months_precip = monthly_precips_cm[3:9]

    if summer_precip >= 0.7 * map_precip:
        p_thresh = 2 * mat + 28
    elif winter_precip >= 0.7 * map_precip:
        p_thresh = 2 * mat
    else:
        p_thresh = 2 * mat + 14

    if map_precip < p_thresh:
        b_type = "BW" if map_precip < 0.5 * p_thresh else "BS"
        h_type = "h" if mat >= 18.0 else "k"
        desc = "Desert (Arid)" if b_type == "BW" else "Steppe (Semi-Arid)"
        zone_detail = "Hot" if h_type == "h" else "Cold"
        return f"{b_type}{h_type}", f"{b_type}{h_type} ({zone_detail} {desc})"

    if thot < 10.0:
        code = "ET" if thot > 0.0 else "EF"
        desc = "Tundra" if thot > 0.0 else "Frost"
        return code, f"{code} ({desc})"

    if tcold >= 18.0:
        pdry = min(monthly_precips_cm)
        if pdry >= 6.0:
            return "Af", "Af (Tropical Rainforest)"
        elif map_precip >= 2.5 * (100 - pdry * 10) / 10:
            return "Am", "Am (Tropical Monsoon)"
        else:
            pdry_is_summer = pdry in s_months_precip
            sub_code = "As" if pdry_is_summer else "Aw"
            desc_detail = "Tropical Wet/Dry (Summer Dry)" if pdry_is_summer else "Tropical Wet/Dry (Winter Dry)"
            return sub_code, f"{sub_code} ({desc_detail})"

    group = "C" if 0.0 <= tcold < 18.0 else ("D" if tcold < 0.0 else "Unclassified")
    if group == "Unclassified":
        return "Unclassified", "Unclassified"

    psdry = min(s_months_precip)
    pswet = max(s_months_precip)
    pwdry = min(w_months_precip)
    pwwet = max(w_months_precip)

    is_s = (psdry < 4.0) and (psdry < pwwet / 3.0)
    is_w = (pwdry < pswet / 10.0)
    season = "s" if is_s and not is_w else ("w" if is_w and not is_s else "f")

    thot_max = max(monthly_temps)
    warm_months_count = sum(1 for t in monthly_temps if t >= 10.0)

    if thot_max >= 22.0:
        temp_third = "a"
    elif warm_months_count >= 4:
        temp_third = "b"
    else:
        temp_third = "d" if (group == "D" and tcold < -38.0) else "c"

    descriptions = {
        "Cfa": "Humid Subtropical", "Cfb": "Oceanic / Maritime Temperate", "Cfc": "Subpolar Oceanic",
        "Csa": "Hot-summer Mediterranean", "Csb": "Warm-summer Mediterranean", "Csc": "Cold-summer Mediterranean",
        "Cwa": "Monsoon-influenced Humid Subtropical", "Cwb": "Subtropical Highland", "Cwc": "Cold Subtropical Highland",
        "Dfa": "Hot-summer Humid Continental", "Dfb": "Warm-summer Humid Continental", "Dfc": "Subarctic / Boreal",
        "Dfd": "Extremely Cold Subarctic", "Dwa": "Monsoon-influenced Hot-summer Humid Continental",
        "Dwb": "Monsoon-influenced Warm-summer Humid Continental", "Dwc": "Monsoon-influenced Subarctic",
        "Dwd": "Monsoon-influenced Extremely Cold Subarctic", "Dsa": "Mediterranean-influenced Hot-summer Humid Continental",
        "Dsb": "Mediterranean-influenced Warm-summer Humid Continental", "Dsc": "Dry-summer Subarctic",
        "Dsd": "Extremely Cold Dry-summer Subarctic",
    }

    code = f"{group}{season}{temp_third}"
    desc = descriptions.get(code, "Temperate/Continental")
    return code, f"{code} ({desc})"


def get_koppen_outlier_expectations(code):
    """Returns rules of thumb for diurnal temperature range (DTR) and outlier bounds by Köppen group."""
    group = code[0] if code else ""
    if group == "B":
        return "High (15-25°C)", "Wide (±10°C to ±15°C from mean due to low humidity)"
    elif group == "A":
        return "Low-Moderate (5-10°C)", "Narrow (±3°C to ±6°C due to high humidity and maritime influence)"
    elif group in ["C", "D"]:
        if "s" in code:
            return "Moderate-High (10-18°C in summer)", "Moderate-Wide (±8°C to ±12°C)"
        return "Moderate (8-14°C)", "Moderate-Wide (±8°C to ±14°C, higher volatility in winter)"
    elif group == "E":
        return "Low-Moderate (4-8°C)", "Wide in winter (±15°C under polar front shifts)"
    return "Variable", "Variable"


def get_grid_elevation(lat, lon):
    """Fetches the digital elevation model (DEM) height for the given grid coordinate."""
    elevation_url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lat, "longitude": lon}
    try:
        response = requests.get(elevation_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            elevations = data.get("elevation", [])
            if elevations:
                return elevations[0]
    except Exception as e:
        print(f"Elevation API error: {e}")
    return None


def get_climate_data(lat, lon, resolved_name, user_elevation=None):
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

            # Calculate Monthly Normals (10-year averages)
            monthly_grouped = (
                df.groupby([df["date"].dt.year, "month"]
            ).agg({"temp": "mean", "temp_max": "mean", "temp_min": "mean", "precip": "sum"})
            .reset_index()
            )

            monthly_normals = (
                monthly_grouped.groupby("month")
                .agg({"temp": "mean", "temp_max": "mean", "temp_min": "mean", "precip": "mean"})
                .reset_index()
            )

            # Calculate 10-Year All-Time Outliers per month
            monthly_outliers = (
                df.groupby("month")
                .agg({"temp_max": "max", "temp_min": "min"})
                .reset_index()
                .rename(columns={"temp_max": "abs_max", "temp_min": "abs_min"})
            )

            monthly_normals = monthly_normals.merge(monthly_outliers, on="month")

            t_list = monthly_normals["temp"].tolist()
            t_max_list = monthly_normals["temp_max"].tolist()
            t_min_list = monthly_normals["temp_min"].tolist()
            abs_max_list = monthly_normals["abs_max"].tolist()
            abs_min_list = monthly_normals["abs_min"].tolist()
            p_list_mm = monthly_normals["precip"].tolist()

            # Elevation Offset Adjustment
            grid_elev = get_grid_elevation(lat, lon)
            elev_diff = 0.0

            if user_elevation is not None and grid_elev is not None:
                elev_diff = user_elevation - grid_elev
                temp_offset = (elev_diff / 1000.0) * -6.5

                # Adjust mean, day max, night min, and extreme outliers
                t_list = [t + temp_offset for t in t_list]
                t_max_list = [t + temp_offset for t in t_max_list]
                t_min_list = [t + temp_offset for t in t_min_list]
                abs_max_list = [t + temp_offset for t in abs_max_list]
                abs_min_list = [t + temp_offset for t in abs_min_list]

            p_list_cm = [p / 10.0 for p in p_list_mm]
            code, koppen_result = classify_koppen(t_list, p_list_cm, lat)
            dtr_expectation, outlier_expectation = get_koppen_outlier_expectations(code)

            print(f"\nClimate normals retrieved successfully for: {resolved_name}")
            if grid_elev is not None:
                print(f"- Grid Elevation (API): {grid_elev:.1f} m")
            if user_elevation is not None and grid_elev is not None:
                print(f"- Custom User Elevation: {user_elevation:.1f} m (Offset Delta: {elev_diff:+.1f} m)")
                print(f"- Applied Temperature Adjustment: {(elev_diff / 1000.0) * -6.5:+.2f}°C")

            print(f"\n--- Köppen-Geiger Climate Classification ---")
            print(f"=> Climate Zone: {koppen_result}")
            print(f"- Typical Diurnal Range Expectation: {dtr_expectation}")
            print(f"- Expected Outlier Volatility: {outlier_expectation}")

            print("\n- Monthly Climate Profile (Mean, Day Avg, Night Avg, Diurnal Range, and 10-Yr Extreme Outliers):")
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            for idx, m_name in enumerate(month_names):
                dtr = t_max_list[idx] - t_min_list[idx]
                print(
                    f"  {m_name}: Mean: {t_list[idx]:.1f}°C | Day Avg: {t_max_list[idx]:.1f}°C | Night Avg: {t_min_list[idx]:.1f}°C | "
                    f"DTR: {dtr:.1f}°C | Outliers: [{abs_min_list[idx]:.1f}°C to {abs_max_list[idx]:.1f}°C] | Precip: {p_list_mm[idx]:.1f} mm"
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

lat_val, lon_val, resolved_name = None, None, ""

if mode_choice == "2":
    coord_input = input("Enter coordinates separated by a comma (e.g., 1.3521, 103.8198): ")
    try:
        lat_str, lon_str = coord_input.split(",")
        lat_val = float(lat_str.strip())
        lon_val = float(lon_str.strip())
        resolved_name = f"Coordinates ({lat_val}, {lon_val})"
    except ValueError:
        print("Invalid coordinate format. Please use numbers separated by a comma.")
else:
    city_input = input("Enter City, Neighborhood, or Landmark: ")
    country_input = input("Enter Country (optional, e.g. Canada, UK, US): ")

    if city_input.strip():
        lat_val, lon_val, resolved_name = resolve_city_to_coords(city_input, country_input)
        if lat_val is None:
            print(f"Could not find coordinates for location: '{city_input}'")
    else:
        print("Please provide at least a city or neighborhood name.")

if lat_val is not None and lon_val is not None:
    user_elev_input = input("Enter custom elevation in meters (press Enter to skip and use default grid elevation): ").strip()
    user_elevation = float(user_elev_input) if user_elev_input else None
    get_climate_data(lat_val, lon_val, resolved_name, user_elevation=user_elevation)
