def classify_heat(lst):

    if lst < 30:
        return "Low Risk"

    elif lst < 35:
        return "Moderate Risk"

    elif lst < 40:
        return "High Risk"

    else:
        return "Extreme Hotspot"


def detect_hotspots(df):

    df["Heat_Risk"] = df["Predicted_LST"].apply(
        classify_heat
    )

    return df