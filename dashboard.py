import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the processed data
df = pd.read_csv("drought_classification_results.csv")  # Replace with your actual file name

# Sidebar filters
st.sidebar.title("Filters")
selected_districts = st.sidebar.multiselect("Select Districts", options=df["NAME_2"].unique(), default=df["NAME_2"].unique())
selected_columns = ["Rainfall_norm", "LST_norm", "NDVI", "NDWI", "MAI"]

# Filter data
filtered_df = df[df["NAME_2"].isin(selected_districts)]

# Title
st.title("Maharashtra District-Level Drought Dashboard")
st.markdown("This dashboard visualizes NDVI, NDWI, LST, Rainfall, and MAI for drought classification.")

# Plot MAI over time
fig1 = px.line(filtered_df, x="date", y="MAI", color="NAME_2", title="Moisture Adequacy Index (MAI) Over Time")
st.plotly_chart(fig1)

# Boxplots for each parameter
st.subheader("Parameter Distribution")
fig2, ax = plt.subplots(1, len(selected_columns), figsize=(20, 5))
for i, col in enumerate(selected_columns):
    sns.boxplot(y=filtered_df[col], ax=ax[i])
    ax[i].set_title(col)
st.pyplot(fig2)

# Classification section
st.subheader("Run Drought Classification")
if st.button("Classify Drought Levels"):
    model_df = filtered_df.copy()
    label_enc = LabelEncoder()
    if "Drought_Class" in model_df.columns:
        model_df["Drought_Class"] = label_enc.fit_transform(model_df["Drought_Class"])
    else:
        st.warning("No 'Drought_Class' column found for classification")

    features = ["Rainfall_norm", "LST_norm", "NDVI", "NDWI", "MAI"]
    model = RandomForestClassifier()
    model.fit(model_df[features], model_df["Drought_Class"])

    model_df["Predicted"] = model.predict(model_df[features])
    model_df["Predicted_Label"] = label_enc.inverse_transform(model_df["Predicted"])

    st.success("Classification complete!")
    st.dataframe(model_df[["NAME_2", "date"] + features + ["Predicted_Label"]])

    # Display prediction counts
    st.subheader("Predicted Drought Classes")
    pred_chart = px.histogram(model_df, x="Predicted_Label", color="Predicted_Label", title="Distribution of Predicted Classes")
    st.plotly_chart(pred_chart)


