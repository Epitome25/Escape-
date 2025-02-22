# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Load the dataset
# file_path = "c:/Users/salik/Downloads/merged_dam_dataset.csv"
# df = pd.read_csv(file_path)

# df["measurement_date"] = pd.to_datetime(df["measurement_date"])
# df = df.drop(columns=["measurement_date"])

# # Define features and target
# X = df.drop(columns=["Risk Level"])
# y = df["Risk Level"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# gbr.fit(X_train, y_train)

# # Predict risk
# y_pred = gbr.predict(X_test)

# # Identify anomalies
# anomalies = X_test[(y_pred < 0) | (y_pred > 2)].copy()
# anomalies["Predicted Risk Level"] = y_pred[(y_pred < 0) | (y_pred > 2)]

# # Streamlit UI
# st.set_page_config(layout="wide")
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select Page", ["Dashboard", "Detailed Columns"])

# if page == "Dashboard":
#     st.title("Dam Safety Authority - Dashboard")
#     col1, col2, col3 = st.columns(3)
#     col4, col5, col6 = st.columns(3)

#     # Show first values from dataset
#     col1.metric("Temperature", df.iloc[0]["Temperature"])
#     col2.metric("Humidity", df.iloc[0]["Humidity"])
#     col3.metric("Water Level", df.iloc[0]["Water Level"])
#     col4.metric("Vibration", df.iloc[0]["Vibration"])
#     col5.metric("Seepage Rate", df.iloc[0]["Seepage Rate"])
#     risk_level = y_pred[0]
#     col6.metric("Predicted Risk Level", round(risk_level, 2), delta_color="inverse")
    
#     if risk_level > -0.0001:
#         st.error("⚠️ Risk Detected!")
    

# elif page == "Detailed Columns":
#     st.title("Detailed Data After Training")
#     df_detailed = X_test.copy()
#     df_detailed["Predicted Risk Level"] = y_pred
#     st.dataframe(df_detailed.style.applymap(lambda x: "background-color: red; color: white" if x > 1 else "", subset=["Predicted Risk Level"]))
    
#     if not anomalies.empty:
#         st.subheader("Anomalies Detected")
#         st.dataframe(anomalies)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
file_path = "c:/Users/salik/Downloads/merged_dam_dataset.csv"
df = pd.read_csv(file_path)

# Convert measurement_date to datetime and drop it
df["measurement_date"] = pd.to_datetime(df["measurement_date"])
df = df.drop(columns=["measurement_date"])

# Define features and target
X = df.drop(columns=["Risk Level"])
y = df["Risk Level"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)

# Predict risk
y_pred = gbr.predict(X_test)

# Identify anomalies
anomalies = X_test[(y_pred < 0) | (y_pred > 2)].copy()
anomalies["Predicted Risk Level"] = y_pred[(y_pred < 0) | (y_pred > 2)]

# Streamlit UI Configuration
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #1e1e1e; color: white; }
    .stMetric { font-size: 24px; }
    .stAlert { font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Detailed Columns"])

if page == "Dashboard":
    st.title("🚧 Dam Safety Authority - Dashboard")

    # First and last values from dataset
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    # Fetch first & last values
    def show_metric(col, label, value1, value2):
        col.metric(label, f"{value1:.2f}", help=f"Last recorded: {value2:.2f}")

    show_metric(col1, "🌡️ Temperature (°C)", df.iloc[-1]["Temperature"], df.iloc[-1]["Temperature"])
    show_metric(col2, "💧 Humidity (%)", df.iloc[-1]["Humidity"], df.iloc[-1]["Humidity"])
    show_metric(col3, "🌊 Water Level (m)", df.iloc[-1]["Water Level"], df.iloc[-1]["Water Level"])
    show_metric(col4, "📉 Vibration", df.iloc[-1]["Vibration"], df.iloc[-1]["Vibration"])
    show_metric(col5, "🔍 Seepage Rate", df.iloc[-1]["Seepage Rate"], df.iloc[-1]["Seepage Rate"])
    
    # Predicted Risk Level
    risk_level = round(y_pred[-1], 2)
    col6.metric("⚠️ Predicted Risk Level", risk_level, delta_color="inverse")

    # Risk Alert
    if risk_level > 1:
        st.error("🚨 **High Risk Detected! Immediate Attention Required!**")
    elif risk_level > 0.5:
        st.warning("⚠️ **Moderate Risk! Monitor Closely.**")
    else:
        st.success("✅ **Safe Conditions.**")

elif page == "Detailed Columns":
    st.title("📊 Detailed Data After Training")
    
    df_detailed = X_test.copy()
    df_detailed["Predicted Risk Level"] = y_pred

    # Highlight risk values
    st.dataframe(
        df_detailed.style.applymap(lambda x: "background-color: red; color: white" if x > 1 else "", subset=["Predicted Risk Level"])
    )

    if not anomalies.empty:
        st.subheader("🚨 Anomalies Detected")
        st.dataframe(anomalies)
