import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap

WORLDCOVER_MAP = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up land",
    60: "Bare_sparse_vegetation",
    80: "Permanent_water_bodies",
    90: "Herbaceous_wetland"
}

EXPECTED_LC_COLS = [
    "LandCover_Bare_sparse_vegetation",
    "LandCover_Built-up land",
    "LandCover_Cropland",
    "LandCover_Grassland",
    "LandCover_Herbaceous_wetland",
    "LandCover_Permanent_water_bodies",
    "LandCover_Shrubland",
    "LandCover_Tree cover"
]

MODEL_FEATURES = ["NDVI", "NDBI", "Elevation", "Population"] + EXPECTED_LC_COLS

HEAT_COLORS = {
    "Low": "#f6c64b",
    "Moderate": "#ff9a22",
    "High": "#f0522d",
    "Very High": "#a91f26"
}


@st.cache_data(show_spinner=False)
def geocode_city(city_query: str):
    """Resolve any city name to (lat, lon, display_name) via OpenStreetMap Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "HeatSense-UrbanAI-Research/1.0"}
    params = {"q": city_query, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not find coordinates for '{city_query}'. Try adding state or country.")
    top = results[0]
    return float(top["lat"]), float(top["lon"]), top.get("display_name", city_query)


def fetch_gee_city_dataset(lat: float, lon: float, half_size_km: float, max_rows: int, gee_project: str):
    """Pull Sentinel-2, Landsat 8/9 LST, SRTM, WorldPop, and ESA WorldCover from GEE (capped at max_rows)."""
    import ee

    try:
        proj = gee_project.strip() if gee_project.strip() else "heatsense-511311"
        ee.Initialize(project=proj)
    except Exception as init_err:
        st.info(f"Re-authenticating GEE session for project: {proj}...")
        ee.Authenticate(force=True)
        ee.Initialize(project=proj)

    delta_lat = half_size_km / 111.0
    delta_lon = half_size_km / (111.0 * max(np.cos(np.radians(lat)), 0.2))
    roi = ee.Geometry.Rectangle([lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2024-03-01", "2025-06-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")

    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(roi)
        .filterDate("2024-03-01", "2025-06-30")
        .filter(ee.Filter.lt("CLOUD_COVER", 15))
        .median()
    )
    lst_c = l8.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")

    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("Elevation")
    # Mosaic (never newest-first): a border-touching foreign tile such as
    # CHN_2020 can otherwise win .sort().first() with zero valid pixels here,
    # fully masking Population and emptying every sampled batch.
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(roi)
        .mosaic()
        .select("population")
        .rename("Population")
    )
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("LandCover_Code")

    lon_lat = ee.Image.pixelLonLat().rename(["Longitude", "Latitude"])
    combined = ee.Image.cat([lon_lat, ndvi, ndbi, lst_c, elevation, pop, landcover]).clip(roi)

    batch_size = 4500
    num_batches = max(1, int(np.ceil(max_rows / batch_size)))
    frames = []

    for b in range(num_batches):
        n_pts = min(batch_size, max_rows - len(frames) * batch_size)
        if n_pts <= 0:
            break
        fc = combined.sample(
            region=roi,
            scale=60,
            numPixels=int(n_pts * 1.15),
            seed=42 + b,
            geometries=False
        ).limit(int(n_pts))
        features = fc.getInfo().get("features", [])
        if features:
            frames.append(pd.DataFrame([f["properties"] for f in features]))

    if not frames:
        raise RuntimeError("No satellite pixels returned from GEE for this region.")

    return pd.concat(frames, ignore_index=True).head(max_rows)


def generate_calibrated_satellite_sample(lat: float, lon: float, half_size_km: float, max_rows: int):
    """Fast calibrated urban satellite generator for instant offline/demo execution."""
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 1000)) % (2**31 - 1))
    delta_lat = half_size_km / 111.0
    delta_lon = half_size_km / (111.0 * max(np.cos(np.radians(lat)), 0.2))

    lats = rng.uniform(lat - delta_lat, lat + delta_lat, max_rows)
    lons = rng.uniform(lon - delta_lon, lon + delta_lon, max_rows)

    dist_norm = np.sqrt(((lats - lat) / delta_lat) ** 2 + ((lons - lon) / delta_lon) ** 2)
    urban_core_weight = np.clip(1.0 - 0.65 * dist_norm + rng.normal(0, 0.18, max_rows), 0, 1)

    ndbi = np.clip(-0.15 + 0.55 * urban_core_weight + rng.normal(0, 0.08, max_rows), -0.6, 0.65)
    ndvi = np.clip(0.52 - 0.44 * urban_core_weight + rng.normal(0, 0.09, max_rows), -0.15, 0.82)
    elevation = np.clip(55 + 25 * np.sin((lats - lat) * 80) + rng.normal(0, 6, max_rows), 5, 800)
    population = np.clip(np.exp(2.2 + 3.1 * urban_core_weight + rng.normal(0, 0.45, max_rows)), 1, 450)

    lc_codes = np.where(
        ndbi > 0.18, 50,
        np.where(
            ndvi > 0.45, 10,
            np.where(
                ndvi > 0.30, 40,
                np.where(ndvi < 0.02, 80, rng.choice([20, 30, 60], size=max_rows))
            )
        )
    )

    base_temp = 38.5 - 0.12 * abs(lat - 23.0)
    lst = (
        base_temp
        + 6.8 * ndbi
        - 5.4 * ndvi
        - 0.015 * (elevation - 50)
        + 0.006 * population
        + np.where(lc_codes == 50, 1.4, np.where(lc_codes == 10, -1.5, np.where(lc_codes == 80, -3.2, 0.2)))
        + rng.normal(0, 1.15, max_rows)
    )

    return pd.DataFrame({
        "Latitude": lats,
        "Longitude": lons,
        "NDVI": ndvi,
        "NDBI": ndbi,
        "LST": lst,
        "Elevation": elevation,
        "Population": population,
        "LandCover_Code": lc_codes
    })


def process_and_train_city_pipeline(raw_df: pd.DataFrame, city_slug: str):
    df = raw_df.dropna(subset=["Latitude", "Longitude", "NDVI", "NDBI", "LST", "Elevation", "Population"]).copy()
    df = df.head(100_000)

    df["LandCover_Class"] = df["LandCover_Code"].astype(int).map(WORLDCOVER_MAP).fillna("Built-up land")
    for lc_name in WORLDCOVER_MAP.values():
        df[f"LandCover_{lc_name}"] = (df["LandCover_Class"] == lc_name).astype(float)

    lat_bins = pd.qcut(df["Latitude"], q=10, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df["Longitude"], q=10, labels=False, duplicates="drop")
    df["Spatial_Block"] = lat_bins * 10 + lon_bins
    prefix = city_slug[:3].upper()
    df["Zone_ID"] = [f"{prefix}_Zone_{int(b)+1:03d}" for b in df["Spatial_Block"]]

    X = df[MODEL_FEATURES]
    y = df["LST"]
    groups = df["Spatial_Block"]

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))

    model = XGBRegressor(
        n_estimators=220,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )

    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])

    cv_r2 = r2_score(y, oof_preds)
    cv_mae = mean_absolute_error(y, oof_preds)
    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))

    model.fit(X, y)
    df["Observed_LST_C"] = df["LST"]
    df["Predicted_LST_C"] = model.predict(X)

    q25, q50, q75 = df["Observed_LST_C"].quantile([0.25, 0.50, 0.75])
    df["Heat_Category"] = pd.cut(
        df["Observed_LST_C"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["Low", "Moderate", "High", "Very High"]
    ).astype(str)

    lst_min, lst_max = df["Predicted_LST_C"].min(), df["Predicted_LST_C"].max()
    pop_min, pop_max = df["Population"].min(), df["Population"].max()
    df["Heat_Hazard_Score"] = (df["Predicted_LST_C"] - lst_min) / max(lst_max - lst_min, 1e-6)
    df["Population_Exposure_Score"] = (df["Population"] - pop_min) / max(pop_max - pop_min, 1e-6)
    df["Heat_Vulnerability_Index"] = 0.60 * df["Heat_Hazard_Score"] + 0.40 * df["Population_Exposure_Score"]

    vq25, vq50, vq75 = df["Heat_Vulnerability_Index"].quantile([0.25, 0.50, 0.75])
    df["Vulnerability_Category"] = pd.cut(
        df["Heat_Vulnerability_Index"],
        bins=[-np.inf, vq25, vq50, vq75, np.inf],
        labels=["Low", "Moderate", "High", "Very High"]
    ).astype(str)

    shap_sample = X.sample(n=min(1200, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(shap_sample)
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Readable_Feature": [f.replace("LandCover_", "Land Cover: ").replace("_", " ") for f in MODEL_FEATURES],
        "Mean_Absolute_SHAP": mean_shap
    }).sort_values("Mean_Absolute_SHAP", ascending=True)

    zone_summary = (
        df.groupby("Zone_ID")
        .agg(
            Center_Latitude=("Latitude", "mean"),
            Center_Longitude=("Longitude", "mean"),
            Mean_Observed_LST_C=("Observed_LST_C", "mean"),
            Mean_Predicted_LST_C=("Predicted_LST_C", "mean"),
            Mean_Vulnerability_Index=("Heat_Vulnerability_Index", "mean"),
            Mean_NDVI=("NDVI", "mean"),
            Mean_NDBI=("NDBI", "mean"),
            Total_Population_Index=("Population", "sum"),
            Dominant_LandCover=("LandCover_Class", lambda s: s.mode().iloc[0] if not s.mode().empty else "Built-up")
        )
        .reset_index()
    )
    top10_hot_zones = zone_summary.sort_values("Mean_Observed_LST_C", ascending=False).head(10).reset_index(drop=True)
    top10_vuln_zones = zone_summary.sort_values("Mean_Vulnerability_Index", ascending=False).head(10).reset_index(drop=True)

    out_dir = Path("cities") / city_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "featured_uhi_v3.csv", index=False)
    top10_hot_zones.to_csv(out_dir / "top10_priority_zones.csv", index=False)
    top10_vuln_zones.to_csv(out_dir / "top10_vulnerability_zones.csv", index=False)
    shap_df.to_csv(out_dir / "shap_importance.csv", index=False)
    joblib.dump(model, out_dir / "xgboost_v3_landcover_model.pkl")

    metrics = {
        "r2": cv_r2,
        "mae": cv_mae,
        "rmse": cv_rmse,
        "rows": len(df),
        "out_dir": str(out_dir)
    }
    return df, model, shap_df, top10_hot_zones, top10_vuln_zones, metrics


def render_multi_city_tab(style_chart_fn=None):
    st.header("🌍 Multi-City Google Earth Engine & AI Trainer")

    st.markdown(
        """
        <div class="heat-note">
            <b>On-Demand City Expansion:</b> Select or type any city name below to geocode its urban core,
            extract satellite layers (capped at ≤100,000 rows), train a city-specific <b>Spatial-CV XGBoost model</b>,
            compute <b>SHAP drivers</b>, and map the <b>Top 10 Priority Hotspots</b>—all saved into <code>cities/&lt;city_name&gt;/</code>
            without overwriting your primary Ahmedabad dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.3, 1.2, 1.1])

    with cfg_col1:
        preset_choice = st.selectbox(
            "Select a Preset City or Custom Search:",
            [
                "Surat, Gujarat, India",
                "Vadodara, Gujarat, India",
                "Rajkot, Gujarat, India",
                "Pune, Maharashtra, India",
                "Jaipur, Rajasthan, India",
                "Bhopal, Madhya Pradesh, India",
                "Hyderabad, Telangana, India",
                "Bengaluru, Karnataka, India",
                "Custom City Name..."
            ]
        )
        if preset_choice == "Custom City Name...":
            city_input = st.text_input("Enter Any City Name:", value="Indore, Madhya Pradesh, India")
        else:
            city_input = preset_choice

    with cfg_col2:
        data_source_mode = st.radio(
            "Satellite Extraction Mode:",
            [
                "🛰️ Live Google Earth Engine API",
                "⚡ Instant Calibrated Satellite Generator (Demo Ready)"
            ]
        )
        gee_project_id = "heatsense-511311"
        if "Live Google Earth Engine" in data_source_mode:
            gee_project_id = st.text_input(
                "Google Cloud / GEE Project ID:",
                value="heatsense-511311",
                placeholder="heatsense-511311"
            )

    with cfg_col3:
        half_box_km = st.slider("Urban Study Radius (km):", 5, 25, 12)
        max_dataset_rows = st.slider(
            "Max Dataset Rows (≤100k):",
            min_value=5_000,
            max_value=100_000,
            value=25_000,
            step=5_000
        )

    train_btn = st.button(
        f"🚀 Fetch Satellite Data & Train AI Model for {city_input}",
        type="primary",
        use_container_width=True
    )

    if train_btn:
        with st.status(f"Running End-to-End GEE + XGBoost Pipeline for **{city_input}**...", expanded=True) as status:
            try:
                st.write("📍 Step 1/4: Geocoding city coordinates via OpenStreetMap...")
                lat, lon, display_name = geocode_city(city_input)
                city_slug = "".join(c if c.isalnum() else "_" for c in city_input.lower()).strip("_")
                st.write(f"✅ Coordinates locked: `{lat:.4f}, {lon:.4f}` ({display_name})")

                st.write(f"🛰️ Step 2/4: Sampling up to **{max_dataset_rows:,}** pixels (Sentinel-2, Landsat 8/9, SRTM, WorldPop, ESA WorldCover)...")
                if "Live Google Earth Engine" in data_source_mode:
                    try:
                        raw_df = fetch_gee_city_dataset(lat, lon, half_box_km, max_dataset_rows, gee_project_id)
                        st.write(f"✅ Extracted `{len(raw_df):,}` live pixels from Google Earth Engine!")
                        data_provenance = "LIVE"
                    except Exception as gee_err:
                        st.warning(f"GEE API Notice ({gee_err}). Automatically switching to calibrated satellite generator.")
                        raw_df = generate_calibrated_satellite_sample(lat, lon, half_box_km, max_dataset_rows)
                        data_provenance = "DEMO"
                else:
                    raw_df = generate_calibrated_satellite_sample(lat, lon, half_box_km, max_dataset_rows)
                    st.write(f"✅ Extracted `{len(raw_df):,}` calibrated urban satellite pixels!")
                    data_provenance = "DEMO"

                st.write("🤖 Step 3/4: Engineering V3 Land-Cover features & running 5-Fold Spatial Block CV XGBoost...")
                df, model, shap_df, top10_hot, top10_vuln, metrics = process_and_train_city_pipeline(raw_df, city_slug)

                st.write("🔍 Step 4/4: Computing TreeSHAP explanations & Top 10 Priority Zone markers...")
                st.session_state["multi_city_state"] = {
                    "city_input": city_input,
                    "display_name": display_name,
                    "lat": lat,
                    "lon": lon,
                    "df": df,
                    "model": model,
                    "shap_df": shap_df,
                    "top10_hot": top10_hot,
                    "top10_vuln": top10_vuln,
                    "metrics": metrics,
                    "data_provenance": data_provenance
                }
                status.update(label=f"✅ Trained & Validated AI Model for {city_input}!", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Pipeline Error", state="error")
                st.error(str(exc))
                return

    if "multi_city_state" in st.session_state:
        state = st.session_state["multi_city_state"]
        df = state["df"]
        model = state["model"]
        shap_df = state["shap_df"]
        top10_hot = state["top10_hot"]
        top10_vuln = state["top10_vuln"]
        metrics = state["metrics"]
        center_lat, center_lon = state["lat"], state["lon"]

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader(f"📊 Trained Model Summary: {state['display_name']}")

        if state.get("data_provenance") == "LIVE":
            st.success("🛰️ LIVE satellite data — metrics below reflect observed pixels for this city.")
        else:
            st.warning("⚠️ DEMO synthetic data — metrics below reflect the calibrated generator formula, not observed pixels.")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Spatial CV R²", f"{metrics['r2']:.3f}")
        k2.metric("MAE (°C)", f"{metrics['mae']:.2f} °C")
        k3.metric("RMSE (°C)", f"{metrics['rmse']:.2f} °C")
        k4.metric("Dataset Rows", f"{metrics['rows']:,}")
        k5.metric("Saved Artifacts", metrics["out_dir"])

        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "🔥 City Hotspots & Top 10 Markers",
            "⚠️ City Vulnerability & Top 10 Markers",
            "📊 City SHAP Drivers",
            "🛰️ City AI Cooling Simulator"
        ])

        with sub_tab1:
            m_mode = st.radio(
                "Map Layer Style:",
                ["Scatter Points", "Gradient Heatmap (LST °C)"],
                horizontal=True,
                key="mc_hot_mode"
            )
            show_pins = st.checkbox("📍 Show Top 10 Hottest Priority Zones (#1 - #10)", value=True, key="mc_hot_pins")

            sample_df = df.sample(n=min(2500, len(df)), random_state=42)
            if "Gradient" in m_mode:
                fig = px.density_map(
                    sample_df, lat="Latitude", lon="Longitude", z="Observed_LST_C",
                    radius=18, zoom=11.2, height=580, color_continuous_scale="YlOrRd",
                    title=f"{state['city_input']} | Land Surface Temperature (°C)"
                )
            else:
                fig = px.scatter_map(
                    sample_df, lat="Latitude", lon="Longitude", color="Heat_Category",
                    color_discrete_map=HEAT_COLORS,
                    category_orders={"Heat_Category": ["Low", "Moderate", "High", "Very High"]},
                    hover_data=["Observed_LST_C", "Predicted_LST_C", "NDVI", "NDBI", "LandCover_Class", "Zone_ID"],
                    zoom=11.2, height=580,
                    title=f"{state['city_input']} | Urban Heat Stress & Top 10 Priority Zones"
                )

            if show_pins:
                fig.add_trace(go.Scattermap(
                    lat=top10_hot["Center_Latitude"],
                    lon=top10_hot["Center_Longitude"],
                    mode="markers",
                    marker=dict(size=18, color="#1a0908", opacity=0.92),
                    hoverinfo="skip",
                    showlegend=False
                ))
                fig.add_trace(go.Scattermap(
                    lat=top10_hot["Center_Latitude"],
                    lon=top10_hot["Center_Longitude"],
                    mode="markers+text",
                    name="📍 Top 10 Hottest Zones",
                    text=[f"#{i+1}" for i in range(len(top10_hot))],
                    textposition="top center",
                    marker=dict(size=11, color="#00e5ff"),
                    hovertext=[
                        f"<b>Rank #{i+1}: {r.Zone_ID}</b><br>Mean LST: {r.Mean_Observed_LST_C:.2f} °C<br>Land Cover: {r.Dominant_LandCover}"
                        for i, r in top10_hot.iterrows()
                    ],
                    hoverinfo="text"
                ))

            fig.update_layout(
                template="plotly_white",
                map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11.2),
                map_style="open-street-map",
                paper_bgcolor="#fffdf9",
                font=dict(color="#1a0908", size=12),
                margin=dict(l=0, r=0, t=45, b=0)
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.dataframe(top10_hot, use_container_width=True, hide_index=True)

        with sub_tab2:
            v_sample = df.sample(n=min(2500, len(df)), random_state=42)
            fig_v = px.density_map(
                v_sample, lat="Latitude", lon="Longitude", z="Heat_Vulnerability_Index",
                radius=18, zoom=11.2, height=580, color_continuous_scale="YlOrRd",
                title=f"{state['city_input']} | Heat Vulnerability Index Surface"
            )
            fig_v.add_trace(go.Scattermap(
                lat=top10_vuln["Center_Latitude"],
                lon=top10_vuln["Center_Longitude"],
                mode="markers+text",
                name="📍 Top 10 Vulnerable Zones",
                text=[f"#{i+1}" for i in range(len(top10_vuln))],
                textposition="top center",
                marker=dict(size=12, color="#00e5ff"),
                hovertext=[
                    f"<b>Rank #{i+1}: {r.Zone_ID}</b><br>HVI: {r.Mean_Vulnerability_Index:.3f}<br>Mean LST: {r.Mean_Predicted_LST_C:.2f} °C"
                    for i, r in top10_vuln.iterrows()
                ],
                hoverinfo="text"
            ))
            fig_v.update_layout(
                template="plotly_white",
                map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11.2),
                map_style="open-street-map",
                paper_bgcolor="#fffdf9",
                font=dict(color="#1a0908", size=12),
                margin=dict(l=0, r=0, t=45, b=0)
            )
            st.plotly_chart(fig_v, use_container_width=True, theme=None)
            st.dataframe(top10_vuln, use_container_width=True, hide_index=True)

        with sub_tab3:
            fig_s = px.bar(
                shap_df, x="Mean_Absolute_SHAP", y="Readable_Feature", orientation="h",
                color="Mean_Absolute_SHAP",
                color_continuous_scale=["#ffcf5c", "#ff8a1c", "#e44720", "#9b1d22"],
                title=f"Global SHAP Drivers for {state['city_input']}"
            )
            fig_s.update_layout(coloraxis_showscale=False, height=440)
            if style_chart_fn:
                fig_s = style_chart_fn(fig_s)
            st.plotly_chart(fig_s, use_container_width=True, theme=None)

        with sub_tab4:
            sc1, sc2, sc3 = st.columns(3)
            g_inc = sc1.slider("🌿 Green Cover Increase (%)", 0, 100, 25, key="mc_g")
            t_inc = sc2.slider("🌳 Tree Plantation (%)", 0, 100, 30, key="mc_t")
            b_red = sc3.slider("🏢 Built-up / Cool Surface Shift (%)", 0, 100, 20, key="mc_b")

            sim_df = df.copy()
            sim_df["NDVI"] = np.clip(sim_df["NDVI"] + g_inc * 0.005 + t_inc * 0.007, -1, 1)
            sim_df["NDBI"] = np.clip(sim_df["NDBI"] - b_red * 0.005 - g_inc * 0.001, -1, 1)
            sim_df["LandCover_Tree cover"] = np.clip(sim_df["LandCover_Tree cover"] + t_inc * 0.004 + g_inc * 0.008, 0, None)
            sim_df["LandCover_Built-up land"] = np.clip(sim_df["LandCover_Built-up land"] - b_red * 0.006, 0, None)
            lc_sum = sim_df[EXPECTED_LC_COLS].sum(axis=1).replace(0, 1)
            sim_df[EXPECTED_LC_COLS] = sim_df[EXPECTED_LC_COLS].div(lc_sum, axis=0)

            sim_df["LST_After"] = model.predict(sim_df[MODEL_FEATURES])
            sim_df["Cooling_C"] = df["Predicted_LST_C"] - sim_df["LST_After"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Mean LST Before", f"{df['Predicted_LST_C'].mean():.2f} °C")
            m2.metric("Mean LST After", f"{sim_df['LST_After'].mean():.2f} °C", delta=f"-{sim_df['Cooling_C'].mean():.2f} °C")
            m3.metric("Max Pixel Cooling", f"{sim_df['Cooling_C'].max():.2f} °C")
