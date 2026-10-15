import pandas as pd


def remove_duplicates(df):
    return df.drop_duplicates()


def handle_missing_values(df):

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def clean_data(df):

    print("Starting Data Cleaning...")

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    print("Data Cleaning Complete.")

    return df