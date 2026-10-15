def create_features(df):

    # Vegetation relative to built-up surface
    df["Green_Built_Ratio"] = (
        df["NDVI"] /
        (df["NDBI"].abs() + 0.01)
    )

    # Population weighted by built-up intensity
    df["Population_Heat_Index"] = (
        df["Population"] *
        df["NDBI"]
    )

    # Elevation weighted by vegetation
    df["Elevation_Cooling_Index"] = (
        df["Elevation"] *
        df["NDVI"]
    )

    return df