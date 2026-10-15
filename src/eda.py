import matplotlib.pyplot as plt
import pandas as pd


def summary_statistics(df):

    print("\nSummary Statistics:\n")

    print(df.describe())


def missing_values(df):

    print("\nMissing Values:\n")

    print(df.isnull().sum())


def correlation_matrix(df):

    print("\nCorrelation Matrix:\n")

    print(df.corr(numeric_only=True))


def plot_histogram(df, column):

    plt.figure(figsize=(8,5))

    df[column].hist(bins=20)

    plt.title(f"{column} Distribution")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.show()