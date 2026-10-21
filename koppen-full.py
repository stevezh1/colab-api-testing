!pip install requests matplotlib numpy ipywidgets

import requests
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Month names for data entry, display, and plotting
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Preset Climate Data Profiles [Temp (°C), Precip (mm)]
PRESETS = {
    "Singapore (Af - Tropical Rainforest)": {
        "lat": 1.3521, "lon": 103.8198, "hemisphere": "Northern",
        "temps": [26.8, 27.1, 27.6, 28.0, 28.3, 28.3, 27.9, 27.9, 27.8, 27.6, 27.0, 26.5],
        "precips": [234.0, 112.8, 170.3, 154.8, 171.2, 130.7, 150.0, 147.0, 156.3, 193.8, 259.9, 318.6]
    },
    "Tokyo, Japan (Cfa - Humid Subtropical)": {
        "lat": 35.6762, "lon": 139.6503, "hemisphere": "Northern",
        "temps": [5.2, 5.7, 8.7, 13.9, 18.2, 21.4, 25.0, 26.4, 22.8, 17.5, 12.1, 7.6],
        "precips": [52.3, 56.1, 117.5, 124.5, 137.8, 167.7, 153.5, 168.2, 209.9, 197.8, 92.5, 51.0]
    },
    "London, UK (Cfb - Oceanic)": {
        "lat": 51.5074, "lon": -0.1278, "hemisphere": "Northern",
        "temps": [5.2, 5.3, 7.6, 9.9, 13.3, 16.5, 18.7, 18.5, 15.7, 12.0, 8.0, 5.5],
        "precips": [55.2, 40.9, 41.6, 43.7, 45.7, 45.2, 45.9, 53.0, 49.7, 65.1, 59.0, 55.2]
    },
    "Cairo, Egypt (BWh - Hot Desert)": {
        "lat": 30.0444, "lon": 31.2357, "hemisphere": "Northern",
        "temps": [13.6, 14.9, 17.6, 21.3, 25.2, 27.8, 28.8, 28.8, 26.7, 24.1, 19.4, 15.1],
        "precips": [5.0, 3.8, 3.8, 1.1, 0.5, 0.1, 0.0, 0.0, 0.0, 0.7, 3.8, 5.9]
    },
    "Sydney, Australia (Cfa - Humid Subtropical)": {
        "lat": -33.8688, "lon": 151.2093, "hemisphere": "Southern",
        "temps": [23.0, 23.0, 21.6, 18.7, 15.8, 13.5, 12.8, 14.1, 16.4, 18.4, 20.0, 22.1],
        "precips": [101.0, 118.0, 129.0, 126.0, 120.0, 132.0, 97.0, 78.0, 68.0, 76.0, 84.0, 77.0]
    },
    "Nuuk, Greenland (ET - Tundra)": {
        "lat": 64.1814, "lon": -51.6941, "hemisphere": "Northern",
        "temps": [-7.5, -7.8, -7.0, -2.2, 2.3, 6.0, 8.6, 8.1, 4.5, -0.3, -3.8, -6.1],
        "precips": [54.0, 51.0, 48.0, 44.0, 53.0, 61.0, 86.0, 89.0, 90.0, 76.0, 74.0, 62.0]
    }
}

KOPPEN_DESCRIPTIONS = {
    "Af": "Tropical Rainforest Climate", "Am": "Tropical Monsoon Climate",
    "Aw": "Tropical Savanna Climate (Dry Winter)", "As": "Tropical Savanna Climate (Dry Summer)",
    "BWh": "Hot Desert Climate", "BWk": "Cold Desert Climate",
    "BSh": "Hot Semi-Arid / Steppe Climate", "BSk": "Cold Semi-Arid / Steppe Climate",
    "Cfa": "Humid Subtropical Climate", "Cfb": "Oceanic / Marine West Coast Climate",
    "Cfc": "Subpolar Oceanic Climate", "Csa": "Hot-summer Mediterranean Climate",
    "Csb": "Warm-summer Mediterranean Climate", "Csc": "Cold-summer Mediterranean Climate",
    "Cwa": "Monsoon-influenced Humid Subtropical Climate", "Cwb": "Monsoon-influenced Subtropical Highland Climate",
    "Cwc": "Monsoon-influenced Subpolar Highland Climate", "Dfa": "Hot-summer Humid Continental Climate",
    "Dfb": "Warm-summer Humid Continental Climate", "Dfc": "Subarctic Climate",
    "Dfd": "Extremely Cold Subarctic Climate", "Dsa": "Mediterranean-influenced Humid Continental Climate",
    "Dsb": "Mediterranean-influenced Warm Continental Climate", "Dsc": "Mediterranean-influenced Subarctic Climate",
    "Dsd": "Mediterranean-influenced Extremely Cold Subarctic Climate", "Dwa": "Monsoon-influenced Humid Continental Climate",
    "Dwb": "Monsoon-influenced Warm Continental Climate", "Dwc": "Monsoon-influenced Subarctic Climate",
    "Dwd": "Monsoon-influenced Extremely Cold Subarctic Climate", "ET": "Tundra Climate", "EF": "Ice Cap Climate"
}


def fetch_open_meteo_climate(lat: float, lon: float) -> tuple[list[float], list[float]]:
    """
    Fetches monthly mean temperature (°C) and precipitation (mm) 
    from Open-Meteo's Climate API (30-year climate normals).
    """
    url = f"https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={lon}&models=EC_Earth3P_HR&daily=temperature_2m_mean,precipitation_sum&start_date=1990-01-01&end_date=2020-12-31"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    daily_temps = data['daily']['temperature_2m_mean']
    daily_precip = data['daily']['precipitation_sum']

    monthly_temp_sum = [0.0] * 12
    monthly_precip_sum = [0.0] * 12
    monthly_counts = [0] * 12

    time_stamps = data['daily']['time']
    for date_str, temp, prec in zip(time_stamps, daily_temps, daily_precip):
        month = int(date_str.split('-')[1]) - 1
        if temp is not None and prec is not None:
            monthly_temp_sum[month] += temp
            monthly_precip_sum[month] += prec
            monthly_counts[month] += 1

    avg_temps = [round(monthly_temp_sum[i] / monthly_counts[i], 1) for i in range(12)]
    avg_precip = [round((monthly_precip_sum[i] / monthly_counts[i]) * 30.4375, 1) for i in range(12)]

    return avg_temps, avg_precip


def classify_koppen(temps: list[float], precips: list[float], is_northern_hemisphere: bool = True) -> dict:
    """Applies the mathematical Köppen-Geiger rules engine."""
    mat = sum(temps) / 12.0
    map_annual = sum(precips)

    if is_northern_hemisphere:
        summer_indices = [3, 4, 5, 6, 7, 8]  # Apr - Sep
        winter_indices = [0, 1, 2, 9, 10, 11] # Oct - Mar
    else:
        summer_indices = [0, 1, 2, 9, 10, 11] # Oct - Mar (SH Summer)
        winter_indices = [3, 4, 5, 6, 7, 8]  # Apr - Sep (SH Winter)

    summer_precip = sum(precips[i] for i in summer_indices)
    winter_precip = sum(precips[i] for i in winter_indices)

    # Aridity Threshold (Pth) calculation
    if summer_precip >= 0.7 * map_annual:
        pth = 20 * mat + 280
    elif winter_precip >= 0.7 * map_annual:
        pth = 20 * mat
    else:
        pth = 20 * mat + 140

    t_max = max(temps)
    t_min = min(temps)
    months_above_10 = sum(1 for t in temps if t >= 10)

    p_smin = min(precips[i] for i in summer_indices)
    p_smax = max(precips[i] for i in summer_indices)
    p_wmin = min(precips[i] for i in winter_indices)
    p_wmax = max(precips[i] for i in winter_indices)

    code = ""

    if t_max < 10:
        code = "EF" if t_max < 0 else "ET"
    elif map_annual < pth:
        group = "BW" if map_annual < 0.5 * pth else "BS"
        sub = "h" if mat >= 18 else "k"
        code = group + sub
    elif t_min >= 18:
        p_dry = min(precips)
        if p_dry >= 60:
            code = "Af"
        elif p_dry >= 100 - (map_annual / 25):
            code = "Am"
        else:
            code = "As" if p_smin < p_wmin else "Aw"
    else:
        is_group_c = t_min > -3

        if p_smin < p_wmin and p_wmax >= 3 * p_smin and p_smin < 40:
            p_code = "s"
        elif p_wmin < p_smin and p_smax >= 10 * p_wmin:
            p_code = "w"
        else:
            p_code = "f"

        if t_max >= 22:
            t_code = "a"
        elif months_above_10 >= 4:
            t_code = "b"
        elif not is_group_c and t_min < -38:
            t_code = "d"
        else:
            t_code = "c"

        code = ("C" if is_group_c else "D") + p_code + t_code

    description = KOPPEN_DESCRIPTIONS.get(code, "Unknown Climate Classification")

    return {
        "code": code,
        "description": description,
        "mat": round(mat, 2),
        "map": round(map_annual, 2),
        "pth": round(pth, 2)
    }


# --- Create Widgets ---
# Top controls
preset_dropdown = widgets.Dropdown(
    options=["-- Select Preset --"] + list(PRESETS.keys()),
    value="Singapore (Af - Tropical Rainforest)",
    description="Preset: ",
    style={'description_width': 'initial'}
)

lat_input = widgets.FloatText(value=1.3521, description="Lat:", layout=widgets.Layout(width='150px'))
lon_input = widgets.FloatText(value=103.8198, description="Lon:", layout=widgets.Layout(width='150px'))
hemisphere_radio = widgets.RadioButtons(
    options=["Northern", "Southern"],
    value="Northern",
    description="Hemisphere:",
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='200px')
)

fetch_button = widgets.Button(description="Fetch API Normals", button_style='primary')

# Input fields for 12 months
temp_widgets = [widgets.FloatText(value=0.0, description=m, layout=widgets.Layout(width='120px')) for m in MONTH_NAMES]
precip_widgets = [widgets.FloatText(value=0.0, description=m, layout=widgets.Layout(width='120px')) for m in MONTH_NAMES]

calc_button = widgets.Button(description="Calculate Classification", button_style='success', layout=widgets.Layout(width='90%', margin='10px 5%'))

# Output areas
result_html = widgets.HTML(value="<h3>Köppen Code: --</h3><p>Description: Select a preset or fetch API data</p>")
plot_output = widgets.Output()

# Define layout grids
input_table_header = widgets.HBox([
    widgets.HTML("<b>Month</b>", layout=widgets.Layout(width='120px', margin='0 0 0 10px')),
    widgets.HTML("<b>Temp (°C)</b>", layout=widgets.Layout(width='120px')),
    widgets.HTML("<b>Precip (mm)</b>", layout=widgets.Layout(width='120px'))
])

input_rows = []
for i in range(12):
    temp_widgets[i].description = ""
    precip_widgets[i].description = ""
    row = widgets.HBox([
        widgets.Label(value=MONTH_NAMES[i], layout=widgets.Layout(width='120px', margin='0 0 0 10px')),
        temp_widgets[i],
        precip_widgets[i]
    ])
    input_rows.append(row)

left_column = widgets.VBox([
    widgets.HTML("<h4>Monthly Climate Data</h4>"),
    input_table_header
] + input_rows + [calc_button], layout=widgets.Layout(border='1px solid #ccc', padding='10px', width='45%'))

right_column = widgets.VBox([
    result_html,
    plot_output
], layout=widgets.Layout(border='1px solid #ccc', padding='10px', width='52%', margin='0 0 0 3%'))

top_controls = widgets.HBox([
    preset_dropdown, lat_input, lon_input, hemisphere_radio, fetch_button
], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='0 0 15px 0', align_items='center'))

app_layout = widgets.VBox([
    widgets.HTML("<h2>Köppen Climate Classification Calculator & API</h2>"),
    top_controls,
    widgets.HBox([left_column, right_column])
])


# --- Event Handlers ---
def plot_walter_lieth(temps, precips):
    with plot_output:
        clear_output(wait=True)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()
        months = np.arange(12)
        
        # Precipitation Bar Chart
        ax1.bar(months, precips, color="#3b82f6", alpha=0.6, width=0.5)
        ax1.set_ylabel("Precipitation (mm)", color="#1d4ed8", fontsize=9)
        ax1.tick_params(axis='y', labelcolor="#1d4ed8")
        ax1.set_xticks(months)
        ax1.set_xticklabels(MONTH_NAMES, fontsize=8)
        ax1.set_ylim(0, max(max(precips) * 1.2, 80))

        # Temperature Line Chart
        ax2.plot(months, temps, color="#ef4444", marker="o", linewidth=2)
        ax2.set_ylabel("Temperature (°C)", color="#b91c1c", fontsize=9)
        ax2.tick_params(axis='y', labelcolor="#b91c1c")
        
        min_t, max_t = min(temps), max(temps)
        ax2.set_ylim(min(min_t - 5, -10), max(max_t + 5, 35))

        plt.tight_layout()
        ax1.grid(True, linestyle="--", alpha=0.4)
        plt.show()


def calculate_classification(b=None):
    try:
        temps = [t.value for t in temp_widgets]
        precips = [p.value for p in precip_widgets]
    except Exception:
        result_html.value = "<h3 style='color:red;'>Invalid Input</h3><p>Please enter valid numbers in all fields.</p>"
        return

    if any(p < 0 for p in precips):
        result_html.value = "<h3 style='color:red;'>Invalid Input</h3><p>Precipitation cannot be negative.</p>"
        return

    is_northern = (hemisphere_radio.value == "Northern")
    res = classify_koppen(temps, precips, is_northern_hemisphere=is_northern)

    result_html.value = f"""
    <h3 style='color: #1e3a8a;'>Köppen Code: {res['code']}</h3>
    <b>Description:</b> {res['description']}<br>
    <span style='font-style: italic; font-size: 11px;'>
        MAT: {res['mat']} °C | MAP: {res['map']} mm | Pth: {res['pth']} mm
    </span>
    """
    plot_walter_lieth(temps, precips)


def load_preset(change):
    selected = change['new']
    if selected in PRESETS:
        data = PRESETS[selected]
        lat_input.value = data["lat"]
        lon_input.value = data["lon"]
        hemisphere_radio.value = data["hemisphere"]
        
        for i in range(12):
            temp_widgets[i].value = data["temps"][i]
            precip_widgets[i].value = data["precips"][i]
        calculate_classification()


def fetch_api(b=None):
    fetch_button.description = "Fetching..."
    fetch_button.disabled = True
    try:
        lat = lat_input.value
        lon = lon_input.value
        hemisphere_radio.value = "Northern" if lat >= 0 else "Southern"
        
        temps, precips = fetch_open_meteo_climate(lat, lon)
        for i in range(12):
            temp_widgets[i].value = temps[i]
            precip_widgets[i].value = precips[i]
        calculate_classification()
    except Exception as e:
        result_html.value = f"<h3 style='color:red;'>API Error</h3><p>Failed to retrieve data: {str(e)}</p>"
    finally:
        fetch_button.description = "Fetch API Normals"
        fetch_button.disabled = False


# --- Wire Events ---
preset_dropdown.observe(load_preset, names='value')
calc_button.on_click(calculate_classification)
fetch_button.on_click(fetch_api)

# Run initial state
display(app_layout)
load_preset({'new': preset_dropdown.value})
