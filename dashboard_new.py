import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px

st.set_page_config(page_title="Maharashtra Drought Dashboard", layout="wide")
st.title("\U0001F3D4 Maharashtra Drought Proneness Dashboard")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("drought_classification_results.csv", parse_dates=['date'])
    df['NAME_2'] = df['NAME_2'].str.lower().str.strip()
    return df

df = load_data()

# Sidebar Filters
unique_dates = sorted(df['date'].dt.date.unique())
selected_date = st.sidebar.selectbox("Select Date", unique_dates)
selected_district = st.sidebar.selectbox("Select District for Time Series", sorted(df['NAME_2'].unique()))
date1 = st.sidebar.selectbox("Compare From Date", unique_dates, index=0)
date2 = st.sidebar.selectbox("Compare To Date", unique_dates, index=len(unique_dates)-1)

# Filter for selected date
filtered_df = df[df['date'].dt.date == selected_date]

# Load Shapefile using GeoPandas
@st.cache_data
def load_geo():
    gdf = gpd.read_file("Maha_districts/Maharashta_Districts.shp").to_crs(epsg=4326)
    gdf["district"] = gdf["NAME_2"].str.lower().str.strip()

    return gdf

geo_df = load_geo()

# Merge Data and GeoJSON
try:
    merged_df = filtered_df.copy()
    merged_df['NAME_2'] = merged_df['NAME_2'].str.lower().str.strip()
    choropleth = px.choropleth(
        merged_df,
        geojson=geo_df.__geo_interface__,
        featureidkey="properties.district",
        locations="NAME_2",
        color="Drought_Proneness",
        color_discrete_map={
            "Low drought prone": "#90ee90",
            "Moderately drought prone": "#ffd700",
            "Severely drought prone": "#ff4500"
        },
        hover_name="NAME_2",
        title=f"Drought Proneness on {selected_date}"
    )
    choropleth.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(choropleth, use_container_width=True)
except Exception as e:
    st.error(f"Error generating map: {e}")

# Time Series Plot for District
district_df = df[df['NAME_2'] == selected_district]
time_series = px.line(
    district_df.sort_values("date"),
    x="date",
    y="Predicted_Class",
    title=f"Predicted Drought Class Over Time for {selected_district.title()}",
    markers=True
)
st.plotly_chart(time_series, use_container_width=True)

# Change Detection Map
compare_df = df[df['date'].dt.date.isin([date1, date2])]
pivot_df = compare_df.pivot_table(index='NAME_2', columns='date', values='Predicted_Class', aggfunc='first').dropna()
pivot_df = pivot_df.reset_index()
pivot_df['Change'] = pivot_df[date2] != pivot_df[date1]
change_df = pivot_df[pivot_df['Change'] == True]
st.subheader("\U0001F4CA Drought Class Changes Between Selected Dates")
st.dataframe(change_df[['NAME_2', date1, date2]])