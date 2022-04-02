from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

import plotly.express as px
import pandas as pd


def plot_na_prop(df: pd.DataFrame):
    """Plots the proportion of missing values for each column

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    Plotly Figure object
        Figure object to be passed on to Streamlit's built-in plotting
    """
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=True)
        missing_data = pd.DataFrame({"Missing Ratio %": na_df})
        fig = px.bar(
            missing_data,
            x="Missing Ratio %",
            title="Proportion of missing values",
            text_auto="d",
        )
        fig.update_layout(title_x=0.5, yaxis_title="")
        fig.update_layout(
            {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",}
        )
    return fig
