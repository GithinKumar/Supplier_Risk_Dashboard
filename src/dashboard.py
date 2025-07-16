import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.express as px

# Load shipment delivery data
delivery_df = pd.read_csv("Data/3.supplier_delivery_dataset.csv")
delivery_df["Value Category"] = delivery_df["Value Category"].fillna("Unknown")  

shipment_categories = ["All"] + sorted(delivery_df["Value Category"].dropna().unique())
shipment_category = st.sidebar.selectbox("Select Value Category", shipment_categories)

# Load the final supplier score data
df = pd.read_csv("Data/6.supplier_score_final.csv")
df["Quarter"] = pd.to_datetime(df["Quarter"])

supplier_options = ["ALL"] + sorted(df["Supplier ID"].unique())
supplier_selected = st.sidebar.selectbox("Select Supplier", supplier_options)

avg_scores = df.groupby("Supplier ID")["Supplier Score"].mean().reset_index()
fig_bar = px.bar(avg_scores, x="Supplier ID", y="Supplier Score", color="Supplier Score", color_continuous_scale="RdYlGn")

if shipment_category != "All":
    filtered_delivery = delivery_df[delivery_df["Value Category"] == shipment_category]
else:
    filtered_delivery = delivery_df

shipment_volume_df = filtered_delivery.groupby("Supplier ID")["Shipment Volume"].sum().reset_index()
shipment_volume_df.rename(columns={"Shipment Volume": "Total Shipment Volume"}, inplace=True)

fig_vol = px.bar(
    shipment_volume_df,
    x="Supplier ID",
    y="Total Shipment Volume",
    color="Total Shipment Volume",
    color_continuous_scale="Blues",
    title="Total Shipment Volume per Supplier",
)

charts = [fig_bar, fig_vol]

if charts:
    # Display charts in rows of up to 3 per row
    rows = [charts[i:i+2] for i in range(0, len(charts), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for idx, chart in enumerate(row):
            with cols[idx]:
                st.plotly_chart(chart, use_container_width=True)

if supplier_selected != "ALL":
    supplier_data = df[df["Supplier ID"] == supplier_selected]
    avg_score = supplier_data["Supplier Score"].mean()
    flagged = "Yes" if avg_score < 75 else "No"

    with st.expander(f"{supplier_selected} - Flagged: {flagged}"):
        st.write(f"**Average Score:** {avg_score:.2f}")
        # Merge supplier data with delivery data to get Value Category
        filtered_supplier_data = supplier_data.merge(
            delivery_df[["Supplier ID", "Value Category"]],
            on="Supplier ID",
            how="left"
        )
        if shipment_category != "All":
            filtered_supplier_data = filtered_supplier_data[filtered_supplier_data["Value Category"] == shipment_category]

        fig_scatter = px.scatter(
            filtered_supplier_data,
            x="Quarter",
            y="Supplier Score",
            color="Value Category" if "Value Category" in filtered_supplier_data.columns else None,
            title=f"{supplier_selected} Quarterly Score",
            trendline="lowess"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Supplier Overview Details")
        overview_df = pd.read_csv("Data/4.supplier_overview.csv")
        filtered_overview = overview_df[
            (overview_df["Supplier ID"] == supplier_selected) &
            ((overview_df["Value Category"] == shipment_category) if shipment_category != "All" else True)
        ]
        st.dataframe(filtered_overview)