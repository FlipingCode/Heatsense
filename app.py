import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from src.multi_city_engine import render_multi_city_tab

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Urban Heat AI Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# SUNSET HEATMAP THEME
# ==================================================

st.markdown(
    """
    <style>
        :root {
            --cream: #fffaf4;
            --panel: #fffdf9;
            --line: #f0ddd0;
            --text: #2f1d1a;

            --maroon: #3b080c;
            --maroon-2: #68131a;
            --red: #b82025;
            --orange-red: #e44720;
            --orange: #ff7a1a;
            --gold: #f7b733;

            --green: #4f7d50;
            --green-soft: #edf6e9;
            --blue: #4d83b8;

            --shadow: 0 8px 22px rgba(93, 35, 19, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(255, 122, 26, 0.10), transparent 18%),
                radial-gradient(circle at 15% 100%, rgba(184, 32, 37, 0.06), transparent 20%),
                var(--cream);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 85% 90%, rgba(255, 122, 26, 0.38), transparent 22%),
                linear-gradient(180deg, #2c0609 0%, #47090e 50%, #741a1b 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] * {
            color: #fffaf6;
        }

        [data-testid="stSidebar"] .stCaption {
            color: #f2cbc0 !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            color: #fffaf6 !important;
            font-weight: 600;
        }

        h1 {
            color: var(--maroon) !important;
            font-size: 2.15rem !important;
            font-weight: 850 !important;
            letter-spacing: -0.7px;
        }

        h2 {
            color: var(--maroon) !important;
            font-weight: 800 !important;
            border-left: 5px solid var(--orange);
            padding-left: 12px;
        }

        h3 {
            color: var(--maroon-2) !important;
            font-weight: 760 !important;
        }

        p, li, label, .stMarkdown {
            color: #4d3832;
        }

        .top-title {
            padding: 8px 0 15px 0;
        }

        .project-title {
            font-size: 2.35rem;
            font-weight: 900;
            letter-spacing: -1px;
            margin: 0;
            background: linear-gradient(90deg, #94191d 0%, #e44720 55%, #ff8a1c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .project-subtitle {
            color: #5e4c46;
            font-size: 1.05rem;
            margin-top: 5px;
        }

        .top-info-card {
            background: rgba(255, 253, 249, 0.94);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 13px 16px;
            box-shadow: 0 5px 16px rgba(93, 35, 19, 0.07);
            min-height: 70px;
        }

        .top-info-label {
            font-size: 0.78rem;
            font-weight: 800;
            color: #9a2a20;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .top-info-value {
            font-size: 0.98rem;
            font-weight: 700;
            color: #35201b;
            margin-top: 3px;
        }

        .section-panel {
            background: rgba(255, 253, 249, 0.93);
            border: 1px solid var(--line);
            border-radius: 13px;
            padding: 17px 18px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }

        .section-title {
            color: var(--maroon);
            font-size: 1.1rem;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .heat-note {
            background: #fff0e9;
            border: 1px solid #f5d2c5;
            border-left: 5px solid var(--orange-red);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 20px 0;
            color: #6b3026;
        }

        .cool-note {
            background: var(--green-soft);
            border: 1px solid #d6e7d0;
            border-left: 5px solid var(--green);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 20px 0;
            color: #35573a;
        }

        .info-note {
            background: #fff8e8;
            border: 1px solid #f3e1b8;
            border-left: 5px solid var(--gold);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 20px 0;
            color: #6a4a18;
        }

        .risk-note {
            background: #fff0ef;
            border: 1px solid #efd2d2;
            border-left: 5px solid var(--red);
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 20px 0;
            color: #6b2e33;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 253, 249, 0.96);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 16px;
            box-shadow: var(--shadow);
        }

        [data-testid="stMetricLabel"] {
            color: #8b5d50 !important;
            font-weight: 750;
        }

        [data-testid="stMetricValue"] {
            color:  #b82025 !important;
            font-weight: 850;
        }

        [data-testid="stDataFrame"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: #fffdf9 !important;
            border-color: #e8cfc2 !important;
        }

        .divider {
            height: 1px;
            background: #efdcd1;
            margin: 26px 0;
        }

        /* Force high-contrast dark labels on all Plotly chart SVG text */
        .js-plotly-plot .plotly text {
            fill: #1a0908 !important;
        }
        .js-plotly-plot .plotly .gtitle,
        .js-plotly-plot .plotly .xtitle,
        .js-plotly-plot .plotly .ytitle,
        .js-plotly-plot .plotly .legendtitletext {
            fill: #3b080c !important;
            font-weight: 700 !important;
        }

        .footer-text {
            text-align: center;
            color: #8a6e65;
            padding: 14px;
            font-size: 0.88rem;
        }

        .sidebar-study-card {
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 11px;
            padding: 14px;
            margin-top: 14px;
            background: rgba(255,255,255,0.06);
        }

        .sidebar-study-title {
            font-size: 0.82rem;
            font-weight: 800;
            color: #ffd9c9;
            margin-bottom: 7px;
        }

        .sidebar-study-value {
            font-size: 0.91rem;
            font-weight: 700;
            color: #fff7f2;
            line-height: 1.45;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# DATA LOADING
# ==================================================

DATA_FOLDER = "outputs/dashboard_data"


@st.cache_data
def load_data():
    hotspot_map = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_hotspot_map_v3.csv"))
    scenario_map = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_scenario_map_v3.csv"))
    priority_zones = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_priority_zones_v3.csv"))
    intervention_zones = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_intervention_zones_v3.csv"))
    shap_importance = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_shap_importance_v3.csv"))
    model_comparison = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_model_comparison.csv"))
    scenario_summary = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_scenario_summary_v3.csv"))
    kpis = pd.read_csv(os.path.join(DATA_FOLDER, "dashboard_kpis_v3.csv"))
    vulnerability_map = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_vulnerability_map_v3.csv"))
    vulnerability_zones = pd.read_csv(os.path.join(
        DATA_FOLDER, "dashboard_vulnerability_zones_v3.csv"))

    return (
        hotspot_map,
        scenario_map,
        priority_zones,
        intervention_zones,
        shap_importance,
        model_comparison,
        scenario_summary,
        kpis,
        vulnerability_map,
        vulnerability_zones
    )


try:
    (
        hotspot_map,
        scenario_map,
        priority_zones,
        intervention_zones,
        shap_importance,
        model_comparison,
        scenario_summary,
        kpis,
        vulnerability_map,
        vulnerability_zones
    ) = load_data()

except FileNotFoundError as error:
    st.error("Dashboard data file not found.")
    st.write("Please run Notebook 13 again before running the dashboard.")
    st.code(str(error))
    st.stop()


# ==================================================
# CONSISTENT COLOURS, LEGENDS AND UNITS
# ==================================================

HEAT_COLOR_MAP = {
    "Low": "#f6c64b",
    "Moderate": "#ff9a22",
    "High": "#f0522d",
    "Very High": "#a91f26"
}

VULNERABILITY_COLOR_MAP = {
    "Low": "#f6c64b",
    "Moderate": "#ff9a22",
    "High": "#f0522d",
    "Very High": "#a91f26"
}

INTERVENTION_COLOR_MAP = {
    "Urban greening": "#4f7d50",
    "Cool/permeable surfaces": "#4d83b8"
}

HEAT_CATEGORY_ORDER = ["Low", "Moderate", "High", "Very High"]

MAP_LEGEND_TITLES = {
    "Heat_Category": "Heat Stress Level",
    "Vulnerability_Category": "Vulnerability Level",
    "Recommended_Intervention": "Recommended Strategy"
}


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_kpi_value(kpi_dict, possible_names, default=0):
    for name in possible_names:
        if name in kpi_dict:
            return kpi_dict[name]
    return default


def clean_map_columns(dataframe, columns_to_keep):
    return [column for column in columns_to_keep if column in dataframe.columns]


def format_hover_labels(hover_columns):
    label_map = {
        "Observed_LST_C": "Observed LST (°C)",
        "Predicted_LST_C": "Predicted LST (°C)",
        "Baseline_Predicted_LST_C": "Baseline Predicted LST (°C)",
        "Greening_Cooling_C": "+20% Green Cover Change (°C)",
        "Cool_Surface_Cooling_C": "Cool/Permeable Surface Change (°C)",
        "Best_Cooling_C": "Best Predicted LST Change (°C)",
        "NDVI": "NDVI",
        "NDBI": "NDBI",
        "Population": "Population Exposure",
        "LandCover_Class": "Land Cover",
        "Zone_ID": "Zone ID",
        "Heat_Hazard_Score": "Heat Hazard Score",
        "Population_Exposure_Score": "Population Exposure Score",
        "Heat_Vulnerability_Index": "Heat Vulnerability Index"
    }

    return {
        column: label_map.get(column, column.replace("_", " "))
        for column in hover_columns
    }


def style_map(fig, zoom=11.2):
    fig.update_layout(
        template="plotly_white",
        map=dict(
            center=dict(lat=23.0225, lon=72.5714),
            zoom=zoom
        ),
        map_style="open-street-map",
        margin={"r": 0, "t": 46, "l": 0, "b": 0},
        paper_bgcolor="#fffdf9",
        plot_bgcolor="#fffdf9",
        font=dict(family="Arial, sans-serif", color="#1a0908", size=13),
        title=dict(font=dict(family="Arial, sans-serif", color="#3b080c", size=16)),
        coloraxis_colorbar=dict(
            tickfont=dict(family="Arial, sans-serif", color="#1a0908", size=12),
            title=dict(font=dict(family="Arial, sans-serif", color="#3b080c", size=13))
        ),
        legend=dict(
            bgcolor="rgba(255,253,249,0.96)",
            bordercolor="#d8bfb2",
            borderwidth=1,
            font=dict(family="Arial, sans-serif", size=12, color="#1a0908"),
            title_font=dict(family="Arial, sans-serif", size=13, color="#3b080c"),
            x=0.99,
            y=0.98,
            xanchor="right",
            yanchor="top"
        )
    )
    return fig


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffdf9",
        plot_bgcolor="#fffdf9",
        font=dict(family="Arial, sans-serif", color="#1a0908", size=13),
        title=dict(font=dict(family="Arial, sans-serif", color="#3b080c", size=16)),
        xaxis=dict(
            gridcolor="#e8d5c8",
            linecolor="#b89f91",
            tickfont=dict(family="Arial, sans-serif", color="#1a0908", size=12),
            title_font=dict(family="Arial, sans-serif", color="#3b080c", size=14)
        ),
        yaxis=dict(
            gridcolor="#e8d5c8",
            linecolor="#b89f91",
            tickfont=dict(family="Arial, sans-serif", color="#1a0908", size=12),
            title_font=dict(family="Arial, sans-serif", color="#3b080c", size=14)
        ),
        legend=dict(
            bgcolor="rgba(255,253,249,0.96)",
            bordercolor="#d8bfb2",
            borderwidth=1,
            font=dict(family="Arial, sans-serif", color="#1a0908", size=12),
            title_font=dict(family="Arial, sans-serif", color="#3b080c", size=13)
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(family="Arial, sans-serif", color="#1a0908", size=12),
            title=dict(font=dict(family="Arial, sans-serif", color="#3b080c", size=13))
        )
    )
    fig.update_traces(
        textfont=dict(family="Arial, sans-serif", color="#1a0908", size=12),
        selector=dict(type="bar")
    )
    return fig



def add_top10_markers(fig, base_df, priority_df=None, label_prefix="Hotspot"):
    """Overlay Top 10 Priority Zone markers (or Top 10 hottest points) onto a Plotly Maplibre figure."""
    top_df = None

    if priority_df is not None and not priority_df.empty:
        p_df = priority_df.copy()
        lat_col = next((c for c in ["Latitude", "Center_Latitude", "lat", "Lat"] if c in p_df.columns), None)
        lon_col = next((c for c in ["Longitude", "Center_Longitude", "lon", "Lon"] if c in p_df.columns), None)

        if lat_col and lon_col:
            p_df["Latitude"] = pd.to_numeric(p_df[lat_col], errors="coerce")
            p_df["Longitude"] = pd.to_numeric(p_df[lon_col], errors="coerce")
            top_df = p_df.dropna(subset=["Latitude", "Longitude"]).head(10).copy()
        elif "Zone_ID" in p_df.columns and "Zone_ID" in base_df.columns:
            centroids = (
                base_df.groupby("Zone_ID")[["Latitude", "Longitude"]]
                .mean()
                .reset_index()
            )
            merged = pd.merge(p_df.head(10), centroids, on="Zone_ID", how="left")
            top_df = merged.dropna(subset=["Latitude", "Longitude"]).copy()

    if top_df is None or top_df.empty:
        sort_col = next(
            (c for c in ["Observed_LST_C", "LST_After", "Predicted_LST_C", "Baseline_Predicted_LST_C", "Heat_Vulnerability_Index"]
             if c in base_df.columns),
            None
        )
        if sort_col:
            top_df = base_df.dropna(subset=["Latitude", "Longitude", sort_col]).nlargest(10, sort_col).copy()
        else:
            return fig

    top_df = top_df.head(10).reset_index(drop=True)
    top_df["Rank"] = [f"#{i+1}" for i in range(len(top_df))]

    zone_col = "Zone_ID" if "Zone_ID" in top_df.columns else None
    temp_col = next(
        (c for c in ["Mean_Observed_LST_C", "Mean_LST_C", "Observed_LST_C", "LST_After", "Predicted_LST_C", "Baseline_Predicted_LST_C", "Mean_Vulnerability_Index", "Heat_Vulnerability_Index"]
         if c in top_df.columns),
        None
    )

    hover_texts = []
    for _, row in top_df.iterrows():
        parts = [f"<b>Rank {row['Rank']} Priority {label_prefix}</b>"]
        if zone_col and pd.notna(row[zone_col]):
            parts.append(f"Zone: {row[zone_col]}")
        if temp_col and pd.notna(row[temp_col]):
            unit = " °C" if "LST" in temp_col else ""
            parts.append(f"{temp_col.replace('_', ' ')}: {float(row[temp_col]):.2f}{unit}")
        parts.append(f"Coords: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
        hover_texts.append("<br>".join(parts))

    # Outer dark halo ring so markers stand out against both red heatmaps and scatter points
    fig.add_trace(
        go.Scattermap(
            lat=top_df["Latitude"],
            lon=top_df["Longitude"],
            mode="markers",
            marker=dict(size=18, color="#1a0908", opacity=0.92),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Inner bright marker + Rank label (#1 - #10)
    fig.add_trace(
        go.Scattermap(
            lat=top_df["Latitude"],
            lon=top_df["Longitude"],
            mode="markers+text",
            name=f"📍 Top 10 {label_prefix}s",
            text=top_df["Rank"],
            textposition="top center",
            textfont=dict(family="Arial, sans-serif", size=12, color="#1a0908"),
            marker=dict(size=11, color="#00e5ff", opacity=1.0),
            hovertext=hover_texts,
            hoverinfo="text"
        )
    )
    return fig


def make_small_clean_map(
    dataframe,
    color_column,
    hover_columns,
    title,
    color_map,
    height=600
):
    """Build a categorical point map or numeric LST density map."""
    map_data = dataframe.copy()
    required = ["Latitude", "Longitude"]
    missing = [column for column in required if column not in map_data.columns]
    if missing:
        st.error("Cannot draw map. Missing coordinate columns: " +
                 ", ".join(missing))
        return go.Figure()

    map_data["Latitude"] = pd.to_numeric(map_data["Latitude"], errors="coerce")
    map_data["Longitude"] = pd.to_numeric(
        map_data["Longitude"], errors="coerce")
    map_data = map_data.dropna(subset=["Latitude", "Longitude"])
    if map_data.empty:
        st.warning("There are no valid coordinates to display.")
        return go.Figure()

    map_mode = st.radio(
        "Map Visualisation Style:",
        ["Scatter Points", "Gradient Heatmap (LST Surface)"],
        horizontal=True,
        key=f"map_mode_{title}"
    )

    # Sample only the map for responsiveness; other dashboard calculations use full data.
    if len(map_data) > 2500:
        map_data = map_data.sample(n=2500, random_state=42)

    category_order = (
        HEAT_CATEGORY_ORDER
        if color_column in ["Heat_Category", "Vulnerability_Category"]
        else None
    )

    if "Gradient" in map_mode:
        if color_column == "Vulnerability_Category" and "Heat_Vulnerability_Index" in map_data.columns:
            numeric_candidates = ["Heat_Vulnerability_Index", "Baseline_Predicted_LST_C", "Predicted_LST_C", "LST"]
        else:
            numeric_candidates = [
                "Observed_LST_C", "Predicted_LST_C",
                "Baseline_Predicted_LST_C", "LST"
            ]
        z_column = next((
            column for column in numeric_candidates
            if column in map_data.columns
            and pd.api.types.is_numeric_dtype(map_data[column])
        ), None)
        if z_column is None:
            st.warning(
                "No numeric LST column is available for a gradient map. Showing scatter points instead.")
            map_mode = "Scatter Points"
        else:
            density_data = map_data.dropna(subset=[z_column])
            if density_data.empty:
                st.warning(
                    f"No valid values are available in {z_column} for the gradient map.")
                return go.Figure()
            fig = px.density_map(
                density_data,
                lat="Latitude",
                lon="Longitude",
                z=z_column,
                radius=18,
                zoom=11.2,
                height=height,
                title=title,
                color_continuous_scale="YlOrRd",
                hover_data=clean_map_columns(
                    density_data, [z_column, "Zone_ID"])
            )
            fig.update_traces(opacity=0.75)

    if map_mode == "Scatter Points":
        if color_column not in map_data.columns:
            st.error(
                f"Cannot draw map: '{color_column}' is missing from the dataset.")
            return go.Figure()
        fig = px.scatter_map(
            map_data,
            lat="Latitude",
            lon="Longitude",
            color=color_column,
            hover_data=clean_map_columns(map_data, hover_columns),
            labels=format_hover_labels(hover_columns),
            zoom=11.2,
            height=height,
            title=title,
            color_discrete_map=color_map,
            category_orders={
                color_column: category_order} if category_order else None
        )
        # Densitymap traces do not support marker; apply this only to scattermap.
        fig.update_traces(
            marker=dict(size=6, opacity=0.64),
            selector=dict(type="scattermap")
        )

    show_top10 = st.checkbox(
        "📍 Show Top 10 Priority / Hottest Markers on Map",
        value=True,
        key=f"top10_toggle_{title}"
    )
    if show_top10:
        is_vuln = color_column == "Vulnerability_Category"
        p_source = vulnerability_zones if is_vuln else priority_zones
        p_label = "Vulnerable Zone" if is_vuln else "Hottest Zone"
        fig = add_top10_markers(fig, dataframe, priority_df=p_source, label_prefix=p_label)

    fig.update_layout(
        legend_title_text=MAP_LEGEND_TITLES.get(
            color_column, color_column.replace("_", " ")
        )
    )
    return style_map(fig, zoom=11.2)


def get_scenario_display_name(scenario_name):
    scenario_lower = str(scenario_name).lower()

    if "green" in scenario_lower or "vegetation" in scenario_lower:
        return "🌿 +20% Green Cover"

    if "cool" in scenario_lower or "permeable" in scenario_lower:
        return "🏙️ Cool / Permeable Surface"

    return str(scenario_name)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown(
    """
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <span style="font-size:24px;">☀️</span>
        <span style="color:white; font-size:20px; font-weight:800;">
            UrbanHeat AI
        </span>
    </div>

    <div style="
        color:#f7d8cf;
        font-size:14px;
        margin-bottom:18px;
    ">
        AI Cooling Intervention Simulator
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        color:white;
        font-size:18px;
        font-weight:700;
        margin-bottom:8px;
    ">
        NAVIGATION
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Select dashboard section",
    [
        "Project Overview",
        "Urban Heat Hotspots",
        "Heat Drivers",
        "Heat Vulnerability",
        "AI Cooling Intervention Simulator",
        "Model Validation",
        "🌍 Multi-City GEE & AI Trainer",
        "About Project",
    ],
)

st.sidebar.markdown(
    """
    <div class="sidebar-study-card">
        <div class="sidebar-study-title">📍 STUDY AREA</div>
        <div class="sidebar-study-value">
            Ahmedabad<br>
            Gujarat, India
        </div>
        <br>
        <div class="sidebar-study-value">
            Year: 2026<br>
            Data: Sentinel-2 + Landsat 8/9<br>
            Model: XGBoost
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
# ==================================================
# TOP HEADER
# ==================================================

header_left, header_right = st.columns([2.1, 1])

with header_left:
    st.markdown(
        """
        <div class="top-title">
            <div class="project-title">URBANHEAT AI</div>
            <div class="project-subtitle">
                Urban Heat Mitigation and Cooling Strategy Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    info_1, info_2 = st.columns([1.7, 0.8])

    with info_1:
        st.markdown(
            """
            <div class="top-info-card">
                <div class="top-info-label">📍 Study Area</div>
                <div class="top-info-value">Ahmedabad, Gujarat</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_2:
        st.markdown(
            """
            <div class="top-info-card">
                <div class="top-info-label">📅 Year</div>
                <div class="top-info-value">2026</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# PROJECT OVERVIEW
# ==================================================

if page == "Project Overview":

    st.markdown(
        """
        <div class="heat-note">
            <b>UrbanHeat AI</b> uses satellite-derived urban indicators and machine learning
            to identify heat-stress hotspots, understand heat drivers, estimate vulnerability,
            and recommend cooling interventions for priority zones.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(

        """
        <div class="section-panel">
            <div class="section-title">🧭 How to Use This Dashboard</div>
            <ol style="margin-bottom:0; padding-left:22px; line-height:1.85;">
                <li><b>Locate heat stress:</b> use Urban Heat Hotspots to identify High and Very High heat zones.</li>
                <li><b>Prioritize people at risk:</b> open Heat Vulnerability to find areas where heat overlaps with population exposure.</li>
                <li><b>Understand why:</b> review Heat Drivers to see which variables most influence predicted land surface temperature.</li>
                <li><b>Compare interventions:</b> use AI Cooling Intervention Simulator to compare +20% green cover and cool/permeable-surface scenarios.</li>
                <li><b>Verify before implementation:</b> use priority-zone tables for field assessment, cost review, and municipal planning.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    kpi_values = dict(zip(kpis["Metric"], kpis["Value"]))

    r2_value = float(get_kpi_value(
        kpi_values,
        ["Dataset V3 XGBoost R2", "Dataset V3 XGBoost R²"],
        0
    ))

    mae_value = float(get_kpi_value(
        kpi_values,
        ["Dataset V3 XGBoost MAE (C)", "Dataset V3 XGBoost MAE"],
        0
    ))

    total_locations = int(float(get_kpi_value(
        kpi_values,
        ["Total Sampled Locations", "Sampled Locations"],
        0
    )))

    priority_count = int(float(get_kpi_value(
        kpi_values,
        ["High Priority Hotspot Zones", "High-Priority Hotspot Zones"],
        0
    )))

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Model R² Score", f"{r2_value:.3f}")
    col2.metric("MAE (°C)", f"{mae_value:.2f} °C")
    col3.metric("Total Locations", f"{total_locations:,}")
    col4.metric("High Priority Zones", f"{priority_count}")

    if "Heat_Category" in hotspot_map.columns:
        high_heat_share = (
            hotspot_map["Heat_Category"]
            .astype(str)
            .str.contains("High", case=False, na=False)
            .mean() * 100
        )
        col5.metric("High Heat Share", f"{high_heat_share:.1f}%")
    else:
        col5.metric("Heat Stress Level", "High")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.45, 0.55])

    with left_col:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🔥 Urban Heat Hotspots</div>', unsafe_allow_html=True)

        heat_options = [
            category for category in HEAT_CATEGORY_ORDER
            if category in hotspot_map["Heat_Category"].dropna().astype(str).unique()
        ]

        selected_heat = st.multiselect(
            "Filter heat-stress categories",
            options=heat_options,
            default=heat_options,
            key="overview_heat_filter"
        )

        filtered_hotspots = hotspot_map[
            hotspot_map["Heat_Category"].astype(str).isin(selected_heat)
        ].copy()

        if len(filtered_hotspots) > 0:
            hover_columns = clean_map_columns(
                filtered_hotspots,
                [
                    "Observed_LST_C",
                    "Predicted_LST_C",
                    "NDVI",
                    "NDBI",
                    "Population",
                    "LandCover_Class",
                    "Zone_ID"
                ]
            )

            fig = make_small_clean_map(
                dataframe=filtered_hotspots,
                color_column="Heat_Category",
                hover_columns=hover_columns,
                title="Urban Heat Hotspots | Land Surface Temperature (°C)",
                color_map=HEAT_COLOR_MAP,
                height=430
            )

            st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig), use_container_width=True, theme=None)

        else:
            st.warning("No locations match the selected heat categories.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Heat Stress Distribution</div>', unsafe_allow_html=True)

        if "Heat_Category" in hotspot_map.columns:
            heat_counts = (
                hotspot_map["Heat_Category"]
                .astype(str)
                .value_counts()
                .reindex(HEAT_CATEGORY_ORDER, fill_value=0)
                .reset_index()
            )
            heat_counts.columns = ["Heat Category", "Count"]

            fig = px.pie(
                heat_counts,
                names="Heat Category",
                values="Count",
                hole=0.64,
                color="Heat Category",
                color_discrete_map=HEAT_COLOR_MAP,
                category_orders={"Heat Category": HEAT_CATEGORY_ORDER}
            )

            fig.update_traces(
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                marker=dict(
                    line=dict(
                        color="#fffaf4",
                        width=2
                    )
                )
            )

            fig.update_layout(
                height=315,
                margin=dict(l=5, r=5, t=10, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#2f1d1a",
                showlegend=True,
                template="plotly_white",
                legend=dict(
                    title=dict(text="Heat Stress Level", font=dict(color="#3b080c", size=13)),
                    orientation="v",
                    x=1.02,
                    y=0.5,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=12, color="#1a0908")
                ),
                annotations=[
                    dict(
                        text="<b>Heat Stress</b><br>Distribution",
                        x=0.5,
                        y=0.5,
                        font=dict(size=12, color="#5d2521"),
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig), use_container_width=True, theme=None)

        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.05, 0.95])

    with col_a:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">📊 Top Heat Drivers (SHAP)</div>', unsafe_allow_html=True)

        shap_plot_data = shap_importance.sort_values(
            by="Mean_Absolute_SHAP",
            ascending=True
        ).copy()

        fig = px.bar(
            shap_plot_data,
            x="Mean_Absolute_SHAP",
            y="Readable_Feature",
            orientation="h",
            color="Mean_Absolute_SHAP",
            color_continuous_scale=["#ffcf5c", "#ff8a1c", "#e44720", "#9b1d22"]
        )

        fig.update_layout(
            coloraxis_showscale=False,
            height=330,
            title="",
            xaxis_title="Mean Absolute SHAP Value"
        )

        st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">❄️ Scenario-Based Predicted LST Change (°C)</div>',
            unsafe_allow_html=True
        )

        if len(scenario_summary) > 0:
            for _, row in scenario_summary.iterrows():
                scenario_name = str(row["Scenario"])
                cooling_value = float(row["Mean_Cooling_C"])
                display_name = get_scenario_display_name(scenario_name)

                if "green" in scenario_name.lower() or "vegetation" in scenario_name.lower():
                    st.markdown(
                        f"""
                        <div class="cool-note">
                            <b>{display_name}</b><br>
                            <span style="font-size:0.88rem;">
                                Predicted LST change from baseline
                            </span><br>
                            <span style="font-size:1.35rem; font-weight:850;">
                                {cooling_value:.3f} °C
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            background:#edf5fd;
                            border:1px solid #d6e5f5;
                            border-left:5px solid #4d83b8;
                            border-radius:10px;
                            padding:14px 16px;
                            margin:10px 0 20px 0;
                            color:#244d7c;
                        ">
                            <b>{display_name}</b><br>
                            <span style="font-size:0.88rem;">
                                Predicted LST change from baseline
                            </span><br>
                            <span style="font-size:1.35rem; font-weight:850;">
                                {cooling_value:.3f} °C
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-note">
            <b>💡 Planning insight:</b> Integrate hotspot intensity, vulnerability level, and scenario-based AI predictions to prioritize high-impact urban heat mitigation strategies.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="risk-note">
            <b>Decision-support disclaimer:</b> Predicted LST changes are AI model–based, scenario-specific planning estimates intended to support comparison and prioritization of urban heat mitigation strategies. Actual outcomes may vary due to local environmental conditions, implementation quality, land availability, engineering feasibility, and site-specific factors. Field verification and detailed technical assessment should be completed before implementation.

        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# URBAN HEAT HOTSPOTS
# ==================================================


elif page == "Urban Heat Hotspots":

    st.header("🔥Urban Heat Hotspots")

    st.markdown(
        """
        <div class="heat-note">
            <b>Purpose:</b> Identify locations with elevated observed land surface
            temperature (LST, °C) and prioritize concentrated hotspot zones for intervention.
        </div>
        """,
        unsafe_allow_html=True
    )

    heat_options = [
        category for category in HEAT_CATEGORY_ORDER
        if category in hotspot_map["Heat_Category"].dropna().astype(str).unique()
    ]

    selected_heat = st.multiselect(
        "Select heat-stress categories",
        options=heat_options,
        default=heat_options,
        key="hotspot_heat_filter"
    )

    filtered_hotspots = hotspot_map[
        hotspot_map["Heat_Category"].astype(str).isin(selected_heat)
    ].copy()

    if len(filtered_hotspots) > 0:
        hover_columns = clean_map_columns(
            filtered_hotspots,
            [
                "Observed_LST_C",
                "Predicted_LST_C",
                "NDVI",
                "NDBI",
                "Population",
                "LandCover_Class",
                "Zone_ID"
            ]
        )

        fig = make_small_clean_map(
            dataframe=filtered_hotspots,
            color_column="Heat_Category",
            hover_columns=hover_columns,
            title="Urban Heat-Stress Distribution | LST (°C)",
            color_map=HEAT_COLOR_MAP,
            height=610
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig), use_container_width=True, theme=None)

    else:
        st.warning("No locations match the selected heat categories.")

    st.subheader("Top High-Priority Hotspot Zones")
    st.dataframe(priority_zones, use_container_width=True, hide_index=True)


# ==================================================
# HEAT DRIVERS
# ==================================================

elif page == "Heat Drivers":

    st.header("📊Heat Drivers and Explainable AI")

    st.markdown(
        """
        <div class="info-note">
            <b>Purpose:</b> Understand which environmental and urban features most
            strongly influence predicted land surface temperature (LST, °C) in the XGBoost model.
        </div>
        """,
        unsafe_allow_html=True
    )

    shap_plot_data = shap_importance.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=True
    ).copy()

    fig = px.bar(
        shap_plot_data,
        x="Mean_Absolute_SHAP",
        y="Readable_Feature",
        orientation="h",
        color="Mean_Absolute_SHAP",
        title="Global SHAP Feature Importance",
        color_continuous_scale=["#ffcf5c", "#ff8a1c", "#e44720", "#9b1d22"]
    )

    fig.update_layout(
        coloraxis_showscale=False,
        height=500,
        xaxis_title="Mean Absolute SHAP Value"
    )

    st.plotly_chart(style_chart(fig), use_container_width=True, theme=None)

    st.markdown(
        """
        <div class="info-note">
            <b>Interpretation:</b> a higher mean absolute SHAP value means that feature
            has a stronger overall influence on model predictions. These results support
            transparency, but do not independently prove causation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        shap_importance[["Readable_Feature", "Mean_Absolute_SHAP"]],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# HEAT VULNERABILITY
# ==================================================

elif page == "Heat Vulnerability":

    st.header("⚠️ Heat Vulnerability")

    st.markdown(
        """
        <div class="risk-note">
            <b>Purpose:</b> Prioritize areas where high predicted heat hazard overlaps with greater population exposure. The hottest location is not necessarily the most vulnerable, making both heat intensity and exposure essential for effective intervention planning.

        </div>
        """,
        unsafe_allow_html=True
    )

    vulnerability_options = [
        category for category in HEAT_CATEGORY_ORDER
        if category in vulnerability_map["Vulnerability_Category"].dropna().astype(str).unique()
    ]

    selected_vulnerability = st.multiselect(
        "Select vulnerability categories",
        options=vulnerability_options,
        default=vulnerability_options
    )

    filtered_vulnerability = vulnerability_map[
        vulnerability_map["Vulnerability_Category"]
        .astype(str)
        .isin(selected_vulnerability)
    ].copy()

    if len(filtered_vulnerability) > 0:
        hover_columns = clean_map_columns(
            filtered_vulnerability,
            [
                "Baseline_Predicted_LST_C",
                "Population",
                "Heat_Hazard_Score",
                "Population_Exposure_Score",
                "Heat_Vulnerability_Index",
                "Zone_ID"
            ]
        )

        fig = make_small_clean_map(
            dataframe=filtered_vulnerability,
            color_column="Vulnerability_Category",
            hover_columns=hover_columns,
            title="Population-Aware Heat Vulnerability | Predicted LST (°C)",
            color_map=VULNERABILITY_COLOR_MAP,
            height=610
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig), use_container_width=True, theme=None)

    else:
        st.warning("No locations match the selected vulnerability categories.")

    st.subheader("Top High-Priority Vulnerability Zones")
    st.dataframe(vulnerability_zones,
                 use_container_width=True, hide_index=True)


# ==================================================
# AI Cooling Intervention Simulator
# ==================================================

elif page == "AI Cooling Intervention Simulator":

    st.header("🛰 AI Cooling Intervention Simulator")

    st.markdown(
        """
        <div class="cool-note">
        This simulator performs <b>real-time AI prediction</b> using the trained
        XGBoost model. Instead of applying fixed temperature reductions, it dynamically modifies satellite-derived urban indicators and
        predicts the updated Land Surface Temperature (LST) through model inference.
        </div>
        """,
        unsafe_allow_html=True,
    )

    MODEL_PATH = Path("outputs/models/xgboost_v3_landcover_model.pkl")
    DATA_PATH = Path("data/processed/featured_uhi_v3.csv")

    MODEL_FEATURES = [
        "NDVI",
        "NDBI",
        "Elevation",
        "Population",
        "LandCover_Bare_sparse_vegetation",
        "LandCover_Built-up land",
        "LandCover_Cropland",
        "LandCover_Grassland",
        "LandCover_Permanent_water_bodies",
        "LandCover_Shrubland",
        "LandCover_Tree cover",
    ]

    @st.cache_resource
    def load_ai_model():
        return joblib.load(MODEL_PATH)

    @st.cache_data
    def load_simulation_dataset():
        return pd.read_csv(DATA_PATH)

    try:
        model = load_ai_model()
        simulation_df = load_simulation_dataset().copy()
    except FileNotFoundError as error:
        st.error(
            "The simulation model or dataset could not be found. Check the expected paths below.")
        st.code(str(error))
        st.info(
            f"Expected model: {MODEL_PATH.as_posix()}\nExpected dataset: {DATA_PATH.as_posix()}")
        st.stop()
    except Exception as error:
        st.error(f"Could not load the simulation model or dataset: {error}")
        st.stop()

    if simulation_df.empty:
        st.error("The simulation dataset is empty. Check the processed CSV file.")
        st.stop()

    if "LandCover_Herbaceous_wetland" not in simulation_df.columns:
        simulation_df["LandCover_Herbaceous_wetland"] = 0.0

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in simulation_df.columns
    ]
    missing_coordinates = [
        column for column in ["Latitude", "Longitude"]
        if column not in simulation_df.columns
    ]

    if missing_features:

        st.error(
            "Required model features are missing:\n\n"
            + "\n".join(missing_features)
        )

        st.stop()

    if missing_coordinates:
        st.error("Required coordinate columns are missing: " +
                 ", ".join(missing_coordinates))
        st.stop()

    st.markdown("---")

    left_col, right_col = st.columns([1, 2])

    with left_col:

        st.subheader("Cooling Interventions")

        green_cover = st.slider(
            "🌿 Green Cover Increase (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
        )

        tree_cover = st.slider(
            "🌳 Tree Plantation (%)",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
        )

        cool_roofs = st.slider(
            "🏠 Cool Roof Adoption (%)",
            min_value=0,
            max_value=100,
            value=15,
            step=1,
        )

        permeable_surface = st.slider(
            "🧱 Permeable Surface (%)",
            min_value=0,
            max_value=100,
            value=15,
            step=1,
        )

        builtup_reduction = st.slider(
            "🏢 Built-up Reduction (%)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

        if st.button(
            "🛰 Run AI Simulation",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["sim_active"] = True
        run_simulation = st.session_state.get("sim_active", False)

    with right_col:

        st.markdown(
            """
        ### Simulation Workflow

        1. Selected interventions modify environmental indicators.

        2. NDVI increases according to vegetation-related actions.

        3. NDBI decreases according to built-up reduction.

        4. Land-cover proportions are updated.

        5. The trained XGBoost model predicts the new LST.

        6. Results are compared against the original scenario.
        """
        )

        st.markdown("---")

    def clip(series, low=None, high=None):
        if low is not None:
            series = np.maximum(series, low)
        if high is not None:
            series = np.minimum(series, high)
        return series

    def normalize_landcover(df):
        cols = [
            "LandCover_Bare_sparse_vegetation",
            "LandCover_Built-up land",
            "LandCover_Cropland",
            "LandCover_Grassland",
            "LandCover_Permanent_water_bodies",
            "LandCover_Shrubland",
            "LandCover_Tree cover",
        ]
        total = df[cols].sum(axis=1).replace(0, 1)
        df[cols] = df[cols].div(total, axis=0)
        return df

    def apply_interventions(df):
        simulated = df.copy()

        # These are hypothetical feature perturbations for scenario comparison,
        # not empirically validated cooling coefficients.
        ndvi_increase = (
            green_cover * 0.0050
            + tree_cover * 0.0070
            + permeable_surface * 0.0020
        )
        simulated["NDVI"] = clip(simulated["NDVI"] + ndvi_increase, -1, 1)

        ndbi_decrease = (
            builtup_reduction * 0.0045
            + green_cover * 0.0008
            + tree_cover * 0.0004
        )
        simulated["NDBI"] = clip(simulated["NDBI"] - ndbi_decrease, -1, 1)

        simulated["LandCover_Tree cover"] = clip(
            simulated["LandCover_Tree cover"] +
            tree_cover * 0.002 + green_cover * 0.010,
            0, None
        )
        simulated["LandCover_Built-up land"] = clip(
            simulated["LandCover_Built-up land"]
            - builtup_reduction * 0.004
            - permeable_surface * 0.002
            - cool_roofs * 0.001,
            0, None
        )
        simulated["LandCover_Cropland"] = clip(
            simulated["LandCover_Cropland"] + green_cover * 0.0005, 0, None
        )
        simulated["LandCover_Grassland"] = clip(
            simulated["LandCover_Grassland"]
            + green_cover * 0.001 + permeable_surface * 0.001,
            0, None
        )
        simulated["LandCover_Shrubland"] = clip(
            simulated["LandCover_Shrubland"] + green_cover * 0.0005, 0, None
        )
        simulated["LandCover_Bare_sparse_vegetation"] = clip(
            simulated["LandCover_Bare_sparse_vegetation"] -
            green_cover * 0.001,
            0, None
        )
        simulated["LandCover_Permanent_water_bodies"] = clip(
            simulated["LandCover_Permanent_water_bodies"], 0, None
        )
        return normalize_landcover(simulated)

    def classify_hotspot(lst):
        if lst >= 42:
            return "Extreme"
        if lst >= 38:
            return "Very High"
        if lst >= 34:
            return "High"
        if lst >= 30:
            return "Moderate"
        return "Low"

    if run_simulation:

        simulated_df = apply_interventions(simulation_df)

        expected_features = (
            model.get_booster().feature_names
            if hasattr(model, "get_booster") and model.get_booster().feature_names
            else list(getattr(model, "feature_names_in_", MODEL_FEATURES))
        )
        for col in expected_features:
            if col not in simulation_df.columns:
                simulation_df[col] = 0.0
            if col not in simulated_df.columns:
                simulated_df[col] = 0.0

        before_prediction = model.predict(
            simulation_df[expected_features]
        )

        after_prediction = model.predict(
            simulated_df[expected_features]
        )

        simulated_df["LST_Before"] = before_prediction
        simulated_df["LST_After"] = after_prediction

        simulated_df["Temperature_Reduction"] = (
            simulated_df["LST_Before"]
            - simulated_df["LST_After"]
        )

        simulated_df["Hotspot_Before"] = (
            simulated_df["LST_Before"]
            .apply(classify_hotspot)
        )
        simulated_df["Hotspot_After"] = (
            simulated_df["LST_After"]
            .apply(classify_hotspot)
        )

        avg_before = simulated_df["LST_Before"].mean()
        avg_after = simulated_df["LST_After"].mean()

        max_before = simulated_df["LST_Before"].max()
        max_after = simulated_df["LST_After"].max()

        reduction = avg_before - avg_after
        HIGH_RISK = ["Very High", "Extreme"]

        hotspots_before = (
            simulated_df["Hotspot_Before"]
            .isin(HIGH_RISK)
            .sum()
        )

        hotspots_after = (
            simulated_df["Hotspot_After"]
            .isin(HIGH_RISK)
            .sum()
        )

        if hotspots_before > 0:

            improvement = (
                (hotspots_before - hotspots_after)
                / hotspots_before
            ) * 100

        else:

            improvement = 0

        # ==========================================================
        # KPI DASHBOARD
        # ==========================================================

        st.markdown("## 📊 AI Simulation Results")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric(
            "Average LST Before",
            f"{avg_before:.2f} °C"
        )

        kpi2.metric(
            "Average LST After",
            f"{avg_after:.2f} °C",
            delta=f"-{reduction:.2f} °C"
        )

        kpi3.metric(
            "Maximum LST Before",
            f"{max_before:.2f} °C"
        )

        kpi4.metric(
            "Maximum LST After",
            f"{max_after:.2f} °C",
            delta=f"-{(max_before-max_after):.2f} °C"
        )

        st.markdown("")

        kpi5, kpi6, kpi7 = st.columns(3)

        kpi5.metric(
            " Priority Hotspots Before",
            int(hotspots_before)
        )

        kpi6.metric(
            " Priority Hotspots After",
            int(hotspots_after)
        )

        kpi7.metric(
            "Priority Hotspot Reduction",
            f"{improvement:.1f}%"
        )

        st.markdown("---")

        # ==========================================================
        # BEFORE VS AFTER COMPARISON
        # ==========================================================
        st.subheader("📈 Average Land Surface Temperature Comparison")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[
                    "Before Intervention",
                    "After Intervention"
                ],
                y=[
                    avg_before,
                    avg_after
                ],
                marker_color=[
                    "#E4572E",
                    "#2E8B57"
                ],
                text=[
                    f"{avg_before:.2f}°C",
                    f"{avg_after:.2f}°C"
                ],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Average LST: %{y:.2f}°C<extra></extra>"
            )
        )

        fig.update_layout(
            height=470,
            template="plotly_white",
            showlegend=False,
            xaxis_title="",
            yaxis_title="Average LST (°C)", yaxis_range=[0, max(avg_before, avg_after) * 1.18],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf9",
            font_color="#2f1d1a"
        )

        fig.update_xaxes(
            gridcolor="#f0e0d7"
        )

        fig.update_yaxes(
            gridcolor="#f0e0d7"
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig),
            use_container_width=True, theme=None,
            config={
                "displaylogo": False,
                "displayModeBar": False
            }
        )

        # ==========================================================
        # TEMPERATURE REDUCTION DISTRIBUTION
        # ==========================================================

        st.subheader("📉 Pixel-wise Temperature Reduction")

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=simulated_df["Temperature_Reduction"],
                nbinsx=40,
                marker_color="#4F7D50",
                hovertemplate="Temperature Reduction: %{x:.2f}°C<br>Pixels: %{y}<extra></extra>"
            )
        )

        fig.update_layout(
            height=420,
            template="plotly_white",
            showlegend=False,
            xaxis_title="Temperature Reduction (°C)",
            yaxis_title="Number of Pixels",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf9",
            font_color="#2f1d1a"
        )

        fig.update_xaxes(
            gridcolor="#f0e0d7"
        )

        fig.update_yaxes(
            gridcolor="#f0e0d7"
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig),
            use_container_width=True, theme=None,
            config={
                "displaylogo": False,
                "displayModeBar": False
            }
        )

        st.markdown("---")

        # ==========================================================
        # AI HOTSPOT MAP (WITH GRADIENT HEATMAP TOGGLE)
        # ==========================================================

        st.subheader("🛰 AI Predicted Heat Hotspots")

        sim_map_mode = st.radio(
            "AI Map Visualisation Mode:",
            [
                "Scatter Points (Hotspot Categories)",
                "Gradient Heatmap (Post-Intervention LST °C)",
                "Cooling Impact Gradient (Temperature Reduction °C)"
            ],
            horizontal=True,
            key="sim_hotspot_map_mode"
        )

        hotspot_colors = {
            "Low": "#4F7D50",
            "Moderate": "#F7B733",
            "High": "#FF7A1A",
            "Very High": "#E44720",
            "Extreme": "#8B0000"
        }

        map_df = simulated_df.copy()
        if len(map_df) > 2500:
            map_df = map_df.sample(n=2500, random_state=42)

        if "Post-Intervention LST" in sim_map_mode:
            fig = px.density_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                z="LST_After",
                radius=18,
                zoom=11.2,
                height=650,
                title="Simulated Land Surface Temperature Surface (°C)",
                color_continuous_scale="YlOrRd",
                hover_data={
                    "Latitude": ":.4f",
                    "Longitude": ":.4f",
                    "LST_Before": ":.2f",
                    "LST_After": ":.2f",
                    "Temperature_Reduction": ":.2f"
                }
            )
            fig.update_traces(opacity=0.78)
        elif "Cooling Impact" in sim_map_mode:
            fig = px.density_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                z="Temperature_Reduction",
                radius=18,
                zoom=11.2,
                height=650,
                title="AI Predicted Cooling Impact Surface (°C Reduction)",
                color_continuous_scale="Tealgrn",
                hover_data={
                    "Latitude": ":.4f",
                    "Longitude": ":.4f",
                    "LST_Before": ":.2f",
                    "LST_After": ":.2f",
                    "Temperature_Reduction": ":.2f"
                }
            )
            fig.update_traces(opacity=0.78)
        else:
            fig = px.scatter_map(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Hotspot_After",
                hover_name="Hotspot_After",
                hover_data={
                    "Latitude": ":.4f",
                    "Longitude": ":.4f",
                    "LST_Before": ":.2f",
                    "LST_After": ":.2f",
                    "Temperature_Reduction": ":.2f"
                },
                color_discrete_map=hotspot_colors,
                category_orders={"Hotspot_After": ["Low", "Moderate", "High", "Very High", "Extreme"]},
                zoom=11.2,
                height=650,
                title="Post-Intervention Hotspot Classification"
            )
            fig.update_traces(
                marker=dict(size=7, opacity=0.72),
                selector=dict(type="scattermap")
            )

        if st.checkbox("📍 Show Top 10 Hottest Post-Intervention Locations", value=True, key="sim_top10_toggle"):
            fig = add_top10_markers(fig, simulated_df, priority_df=None, label_prefix="Simulated Hotspot")
        fig.update_layout(legend_title_text="Hotspot Category")
        fig = style_map(fig, zoom=11.2)

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        st.markdown("---")

        # ==========================================================
        # UPDATED HOTSPOT TABLE
        # ==========================================================

        st.subheader("🔥 Updated Hotspot Classification")

        hotspot_table = (
            simulated_df[
                [
                    "Latitude",
                    "Longitude",
                    "LST_Before",
                    "LST_After",
                    "Temperature_Reduction",
                    "Hotspot_Before",
                    "Hotspot_After"
                ]
            ]
            .sort_values(
                by="LST_After",
                ascending=False
            )
            .reset_index(drop=True)
        )

        hotspot_table.rename(
            columns={
                "LST_Before": "Before (°C)",
                "LST_After": "After (°C)",
                "Temperature_Reduction": "Cooling (°C)",
                "Hotspot_Before": "Category Before",
                "Hotspot_After": "Category After"
            },
            inplace=True
        )

        st.dataframe(
            hotspot_table,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ==========================================================
        # HOTSPOT SUMMARY
        # ==========================================================

        st.subheader("📍 Hotspot Category Summary")

        before_summary = (
            simulated_df["Hotspot_Before"]
            .value_counts()
            .rename("Before")
        )

        after_summary = (
            simulated_df["Hotspot_After"]
            .value_counts()
            .rename("After")
        )

        hotspot_summary = (
            pd.concat(
                [
                    before_summary,
                    after_summary
                ],
                axis=1
            )
            .fillna(0)
            .astype(int)
        )

        hotspot_summary = hotspot_summary.reindex(
            [
                "Extreme",
                "Very High",
                "High",
                "Moderate",
                "Low"
            ],
            fill_value=0
        )

        hotspot_summary = hotspot_summary.reset_index()

        hotspot_summary.columns = [
            "Hotspot Category",
            "Before",
            "After"
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Before",
                x=hotspot_summary["Hotspot Category"],
                y=hotspot_summary["Before"],
                marker_color="#E4572E",
                hovertemplate="<b>Before</b><br>%{x}: %{y}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Bar(
                name="After",
                x=hotspot_summary["Hotspot Category"],
                y=hotspot_summary["After"],
                marker_color="#4F7D50",
                hovertemplate="<b>After</b><br>%{x}: %{y}<extra></extra>"
            )
        )

        fig.update_layout(
            barmode="group",
            height=430,
            template="plotly_white",
            xaxis_title="",
            yaxis_title="Number of Pixels",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf9",
            font_color="#2f1d1a",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0.5,
                xanchor="center"
            )
        )

        fig.update_xaxes(
            gridcolor="#f0e0d7"
        )

        fig.update_yaxes(
            gridcolor="#f0e0d7"
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig),
            use_container_width=True, theme=None,
            config={
                "displaylogo": False,
                "displayModeBar": False
            }
        )

        st.markdown("---")
        # ==========================================================
        # MUNICIPAL RECOMMENDATION ENGINE
        # ==========================================================

        st.subheader("🏛 AI Municipal Recommendation")

        recommendations = []

        if avg_after >= 40:
            recommendations.append(
                "🔥 Immediate intervention is recommended for high-temperature urban zones."
            )

        if tree_cover < 30:
            recommendations.append(
                "🌳 Increase urban tree plantation to improve canopy density and shading."
            )

        if green_cover < 30:
            recommendations.append(
                "🌿 Develop additional green parks, roadside plantations, and urban forests."
            )

        if cool_roofs < 25:
            recommendations.append(
                "🏠 Promote cool roof technology for public and commercial buildings."
            )

        if permeable_surface < 25:
            recommendations.append(
                "🧱 Replace impervious pavements with permeable materials wherever feasible."
            )

        if builtup_reduction < 20:
            recommendations.append(
                "🏢 Encourage redevelopment strategies that reduce excessive built-up density."
            )

        if reduction >= 2:
            recommendations.append(
                "❄ The selected intervention scenario shows strong cooling potential and should be prioritized."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "✅ The current intervention strategy is balanced. Continue monitoring urban heat conditions periodically."
            )

        st.success("\n\n".join(recommendations))

        st.markdown("---")

        # ==========================================================
        # SIMULATION SUMMARY
        # ==========================================================

        st.subheader("📋 Simulation Summary")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Average LST Before (°C)",
                    "Average LST After (°C)",
                    "Temperature Reduction (°C)",
                    "Maximum LST Before (°C)",
                    "Maximum LST After (°C)",
                    "Priority Hotspots Before",
                    "Priority Hotspots After",
                    "Overall Improvement (%)"
                ],
                "Value": [
                    round(avg_before, 2),
                    round(avg_after, 2),
                    round(reduction, 2),
                    round(max_before, 2),
                    round(max_after, 2),
                    int(hotspots_before),
                    int(hotspots_after),
                    round(improvement, 2)
                ]
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ==========================================================
        # DOWNLOAD RESULTS
        # ==========================================================

        st.subheader("📥 Download AI Simulation Results")

        download_df = simulated_df.copy()

        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📄 Download Simulation Results (CSV)",
            data=csv,
            file_name="UrbanHeat_AI_Simulation.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")

        # ==========================================================
        # DISCLAIMER
        # ==========================================================

        st.info(
            """
###  AI Decision Support Disclaimer

Predictions are generated using the trained **XGBoost Urban Heat Prediction Model** and are intended to support urban heat mitigation planning.

Actual cooling performance may vary depending on:

• Quality of intervention implementation

• Local environmental and climatic conditions

• Land availability and urban constraints

• Engineering feasibility

• Long-term maintenance practices

• Site-specific characteristics

This simulator is designed as an **AI-assisted decision-support tool** for municipal authorities, planners, and researchers. Results should be interpreted alongside field observations, engineering studies, and local planning guidelines before implementation.

"""
        )

    else:

        st.info(
            """
Adjust the intervention sliders and click

### 🛰 Run AI Simulation

to generate a real-time AI prediction using the trained XGBoost model.
"""
        )
        # ==========================================================
        # ADDITIONAL ANALYTICS
        # ==========================================================

        st.markdown("---")

        st.subheader("📈 Intervention Effectiveness")

        effectiveness_df = pd.DataFrame(
            {
                "Intervention": [
                    "Green Cover",
                    "Tree Plantation",
                    "Cool Roof",
                    "Permeable Surface",
                    "Built-up Reduction"
                ],
                "Selected (%)": [
                    green_cover,
                    tree_cover,
                    cool_roofs,
                    permeable_surface,
                    builtup_reduction
                ]
            }
        )

        colors = [
            "#4F7D50",
            "#6A994E",
            "#FF7A1A",
            "#F7B733",
            "#E44720"
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=effectiveness_df["Intervention"],
                y=effectiveness_df["Selected (%)"],
                marker_color=colors,
                text=effectiveness_df["Selected (%)"],
                texttemplate="%{text}%",
                textposition="outside",
                hovertemplate="%{x}<br>%{y}%<extra></extra>"
            )
        )

        fig.update_layout(
            height=420,
            template="plotly_white",
            showlegend=False,
            xaxis_title="",
            yaxis_title="Selected Percentage (%)", yaxis_range=[0, 115],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf9",
            font_color="#2f1d1a"
        )

        fig.update_xaxes(
            gridcolor="#f0e0d7"
        )

        fig.update_yaxes(
            gridcolor="#f0e0d7"
        )

        st.plotly_chart(style_chart(fig) if 'scattermap' not in str(fig.data) and 'densitymap' not in str(fig.data) else style_map(fig),
            use_container_width=True
        )

        # ==========================================================
        # AI INSIGHTS
        # ==========================================================

        st.subheader("🤖 AI Insights")

        insight_1 = (
            "Higher NDVI values generally reduce predicted urban heat."
        )

        insight_2 = (
            "Reducing built-up intensity lowers the predicted LST in densely urbanized regions."
        )

        insight_3 = (
            "Tree plantation contributes both through increased NDVI and increased tree-cover proportion."
        )

        insight_4 = (
            "Combining multiple interventions usually produces greater cooling benefits than implementing a single intervention."
        )

        st.markdown(
            f"""
<div class="cool-note">

<b>Key AI Observations</b>

• {insight_1}

• {insight_2}

• {insight_3}

• {insight_4}

</div>
""",
            unsafe_allow_html=True,
        )

    # ==========================================================
    # END OF AI COOLING INTERVENTION SIMULATOR
    # ==========================================================

# ==================================================
# MODEL VALIDATION
# ==================================================

elif page == "Model Validation":

    st.header("📈Model Validation")

    st.markdown(
        """
        <div class="info-note">
            <b>Validation approach:</b> Dataset V2 and Dataset V3 are evaluated using
            spatial block cross-validation to reduce spatial leakage and provide a more
            realistic estimate of model performance in unseen locations.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(model_comparison, use_container_width=True, hide_index=True)

    metric_options = [
        column for column in model_comparison.columns
        if column not in ["Dataset", "Model"]
        and pd.api.types.is_numeric_dtype(model_comparison[column])
    ]

    if metric_options:
        selected_metric = st.selectbox(
            "Select performance metric",
            metric_options
        )

        fig = px.bar(
            model_comparison,
            x="Dataset",
            y=selected_metric,
            color="Dataset",
            text_auto=".3f",
            title=f"Dataset V2 vs Dataset V3: {selected_metric}",
            color_discrete_sequence=[
                "#b82025",
                "#ff7a1a",
                "#4f7d50"
            ]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.info(
        "Latitude and Longitude are used only for mapping and spatial grouping. "
        "They are excluded from model features to prevent target leakage and spatial leakage."
    )

# ===================================================
# ABOUT PROJECT
# ==================================================
elif page == "🌍 Multi-City GEE & AI Trainer":
    render_multi_city_tab(style_chart_fn=style_chart)

elif page == "About Project":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        <div class="section-panel">
            <div class="section-title">Urban Heat Detection, Prediction & AI Cooling Intervention Planning</div>
            This project develops an AI/ML-powered geospatial decision-support system for
            Ahmedabad, Gujarat. It integrates satellite-derived NDVI, NDBI, LST,
            elevation, population, and land-cover information to detect heat-stress hotspots,
            predict urban heat, explain model drivers, identify vulnerable zones, and simulate
            cooling interventions.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Inputs")
        st.markdown(
            """
            - Sentinel-2 SR Harmonized  
            - Landsat 8/9 Collection 2 Level 2  
            - SRTM Elevation  
            - WorldPop India 2020  
            - Google Earth Engine  
            """
        )

    with col2:
        st.subheader("Key Outputs")
        st.markdown(
            """
            - Urban heat hotspot map  
            - XGBoost LST prediction model  
            - Spatial cross-validation results  
            - SHAP feature importance  
            - Cooling intervention comparison  
            - Heat vulnerability priority zones  
            """
        )

    st.markdown(
        """
        <div class="risk-note">
            <b>Limitations:</b> Satellite-derived LST is not the same as near-surface air
            temperature. Population is used as an exposure proxy, and scenario outputs are
            decision-support estimates. Field validation and local feasibility assessment are
            required before implementation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">
        UrbanHeat AI | Urban Heat Detection, Prediction & AI Cooling Intervention Planning
    </div>
    """,
    unsafe_allow_html=True
)
