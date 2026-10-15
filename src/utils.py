import pandas as pd


def load_data(path):

    return pd.read_csv(path)


def save_data(df, path):

    df.to_csv(path, index=False)

    print(f"File saved to {path}")