# frontend/visualization.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Page config
st.set_page_config(page_title="📊 Data Visualization", page_icon="📊", layout="wide")

st.title("📊 California Housing Data Visualization")
st.write("Explore the dataset through different visualizations.")

# Load dataset
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame

# Show data info
st.subheader("Dataset Overview")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# Sidebar for plot selection
st.sidebar.header("Visualization Options")
plot_type = st.sidebar.selectbox(
    "Choose a plot type",
    ("Histogram of Prices", "Scatter Plot (Income vs Price)", "Average Price by House Age", "Population Distribution")
)

# Generate plots
if plot_type == "Histogram of Prices":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['MedHouseVal'], bins=30, kde=True, ax=ax, color="skyblue")
    ax.set_title("Distribution of House Prices")
    ax.set_xlabel("Median House Value (in 100,000s)")
    st.pyplot(fig)

elif plot_type == "Scatter Plot (Income vs Price)":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.5)
    ax.set_title("Median Income vs House Price")
    ax.set_xlabel("Median Income")
    ax.set_ylabel("Median House Value (in 100,000s)")
    st.pyplot(fig)

elif plot_type == "Average Price by House Age":
    fig, ax = plt.subplots(figsize=(8, 5))
    age_bins = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 40, 50])
    avg_prices = df.groupby(age_bins)['MedHouseVal'].mean()
    avg_prices.plot(kind='bar', color='orange', ax=ax)
    ax.set_title("Average House Price by Age Group")
    ax.set_ylabel("Median House Value (in 100,000s)")
    st.pyplot(fig)

elif plot_type == "Population Distribution":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Population'], bins=50, kde=True, color="green")
    ax.set_title("Population Distribution")
    ax.set_xlabel("Population")
    st.pyplot(fig)
