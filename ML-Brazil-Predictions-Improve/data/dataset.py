import pandas as pd
import numpy as np


def load_dataset():
    dataset = pd.read_csv("Vitoria.csv")

    cities = ['Vitória']
    df_filtered = dataset[dataset.city.isin(cities)]

    df_filtered = df_filtered.replace(np.nan, 0)

    print(f"NaN: {df_filtered.isnull().values.any()}")

    return df_filtered
