import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Malaria Data Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.title("Malaria Surveillance & Cases Overview")
st.write(
    "Sample dashboard using Streamlit and Matplotlib. Replace the sample data "
    "below with your CSV or dataset."
)

# Sample dataset (replace with pd.read_csv("your_file.csv") if needed)
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Reported_Cases": [120, 145, 190, 260, 340, 410],
    "Recovered_Cases": [105, 130, 170, 235, 310, 375],
}
df = pd.DataFrame(data)

# Sidebar controls
min_cases = st.sidebar.slider(
    "Filter months with minimum reported cases",
    min_value=0,
    max_value=int(df["Reported_Cases"].max()),
    value=100,
    step=10,
)

filtered_df = df[df["Reported_Cases"] >= min_cases]

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Months Shown", len(filtered_df))
col2.metric("Total Reported Cases", int(filtered_df["Reported_Cases"].sum()))
col3.metric("Total Recovered", int(filtered_df["Recovered_Cases"].sum()))

st.subheader("Data Table")
st.dataframe(filtered_df, use_container_width=True)

st.subheader("Monthly Trends")
if filtered_df.empty:
    st.warning("No records match the selected minimum cases filter.")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(filtered_df["Month"]))
    width = 0.35

    ax.bar(
        x - width / 2,
        filtered_df["Reported_Cases"],
        width,
        label="Reported Cases",
    )
    ax.bar(
        x + width / 2,
        filtered_df["Recovered_Cases"],
        width,
        label="Recovered Cases",
    )

    ax.set_title("Reported vs. Recovered Cases by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Cases")
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_df["Month"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
