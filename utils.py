import streamlit as st
import pandas as pd


@st.cache
def load_and_preprocess_data():
    df = pd.read_csv(
        "Data/TestData.csv",  # Update the CSV file name
    )

    # Remove nans values
    df = df.dropna()

    # Get unique entries in the dataset of users
    users = df["Account ID"].unique()

    # Create a categorical type for users. User ordered to ensure reproducibility
    user_cat = pd.CategoricalDtype(categories=sorted(users), ordered=True)

    # Transform and get the indexes of the columns
    user_idx = df["Account ID"].astype(user_cat).cat.codes

    # Add the categorical index to the starting dataframe
    df["UserIndex"] = user_idx

    return df, user_idx