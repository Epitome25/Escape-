# import streamlit as st
# import pandas as pd
# import time
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# # Load the dataset
# file_path = "c:/Users/salik/Downloads/merged_dam_dataset.csv"
# df = pd.read_csv(file_path)

# # Convert measurement_date to datetime and drop it (if exists)
# if "measurement_date" in df.columns:
#     df["measurement_date"] = pd.to_datetime(df["measurement_date"])
#     df.drop(columns=["measurement_date"], inplace=True)

# # Fill missing values
# df.fillna(df.mean(), inplace=True)

# # Define features and target variable
# X = df.drop(columns=["Risk Level"])
# y = df["Risk Level"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# gbr.fit(X_train, y_train)

# # Predict
# y_pred = gbr.predict(X_test)

# # Model evaluation
# r2 = r2_score(y_test, y_pred)

# # Streamlit UI
# st.set_page_config(page_title="Dam Risk Monitoring", layout="wide")

# st.title("🌊 Dam Risk Monitoring Dashboard")

# # Show R² Score (Accuracy)
# st.metric(label="📊 Model Accuracy (R² Score)", value=f"{r2:.2%}")

# # Scrolling Animation for Dataset Attributes
# st.header("📜 Dataset Attributes (Scrolling)")

# columns = list(X.columns)
# placeholder = st.empty()

# for i in range(10):  # Loop to repeat scrolling
#     for col in columns:
#         placeholder.markdown(f"### 🔹 {col}")  # Display feature name
#         time.sleep(0.7)  # Pause before showing the next attribute

# # Show dataset sample
# st.subheader("📋 Dataset Preview")
# st.dataframe(df.head())

# st.success("✅ Dashboard Loaded Successfully!")

# import streamlit as st
# import pandas as pd
# import time
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# # Load the dataset
# file_path = "c:/Users/salik/Downloads/merged_dam_dataset.csv"
# df = pd.read_csv(file_path)

# # Convert measurement_date to datetime and drop it (if exists)
# if "measurement_date" in df.columns:
#     df["measurement_date"] = pd.to_datetime(df["measurement_date"])
#     df.drop(columns=["measurement_date"], inplace=True)

# # Fill missing values
# df.fillna(df.mean(), inplace=True)

# # Define features and target variable
# X = df.drop(columns=["Risk Level"])
# y = df["Risk Level"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# gbr.fit(X_train, y_train)

# # Predict
# y_pred = gbr.predict(X_test)

# # Model evaluation
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Identify anomalies where predicted risk level is outside the range [0,2]
# X_test["Predicted Risk Level"] = y_pred
# anomalies = X_test[(y_pred < 0) | (y_pred > 2)].copy()

# # Alarm if high risk detected
# high_risk_threshold = -0.0001
# alarm_triggered = any(y_pred > high_risk_threshold)

# # Streamlit UI
# st.set_page_config(page_title="Dam Risk Monitoring", layout="wide")

# st.title("🌊 Dam Risk Monitoring Dashboard")

# # Show model accuracy
# #st.metric(label="📊 Model Accuracy (R² Score)", value=f"{r2:.2%}")

# # Show high-risk alert if applicable
# if alarm_triggered:
#     st.error("🚨 HIGH RISK DETECTED! Immediate Attention Required! 🚨")

# # Scrolling Animation for Dataset Attributes
# st.header("📜 Dataset Attributes")

# columns = list(X.columns)
# placeholder = st.empty()

# # for i in range(2):  # Loop to repeat scrolling twice
# #     for col in columns:
# #         placeholder.markdown(f"### 🔹 {col}")  # Display feature name
# #         time.sleep(0.7)  # Pause before showing the next attribute

# # Display anomalies with risk levels
# st.subheader("⚠️ Anomalies Detected")
# if not anomalies.empty:
#     st.dataframe(anomalies)
# else:
#     st.success("✅ No Anomalies Detected")

# # Show dataset sample with predictions
# st.subheader("📋 Predicted Risk Levels")
# st.dataframe(X_test.head())
# st.success("✅ Dashboard Loaded Successfully!")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
def load_data():
    file_path = "c:/Users/salik/Downloads/merged_dam_dataset.csv"  # Ensure the file is in the working directory
    df = pd.read_csv(file_path)
    df["measurement_date"] = pd.to_datetime(df["measurement_date"])
    df.drop(columns=["measurement_date"], inplace=True)
    return df

# Train model
def train_model(df):
    X = df.drop(columns=["Risk Level"])
    y = df["Risk Level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    X_test["Predicted Risk Level"] = y_pred
    anomalies = X_test[(y_pred < 0) | (y_pred > 2)].copy()
    
    return model, mse, r2, X_test, anomalies

# Load and process data
df = load_data()
model, mse, r2, predictions, anomalies = train_model(df)

# Streamlit UI
st.set_page_config(page_title="Dam Safety Dashboard", layout="wide")
st.title("🌊 Dam Safety Department")

menu = st.sidebar.radio("Navigation", ["Dashboard", "Detailed Columns"])

if menu == "Dashboard":
    st.subheader("Dashboard Overview")
    
    cols = st.columns(4)
    attributes = list(df.columns[:-1])
    values = df.iloc[0, :-1]
    
    for i, col in enumerate(cols * (len(attributes) // 4 + 1)):
        if i < len(attributes):
            risk_style = "background-color: red;" if attributes[i] == "Risk Level" and values[i] > 2 else ""
            col.markdown(f"""
                <div style="text-align: center; padding: 20px; {risk_style}">
                    <h4>{attributes[i]}</h4>
                    <p><b>{values[i]}</b></p>
                </div>
            """, unsafe_allow_html=True)

    st.metric(label="Model Accuracy (R²)", value=f"{r2:.2%}")

elif menu == "Detailed Columns":
    st.subheader("Detailed Data View")
    
    st.write("### Processed Data with Predicted Risk Levels")
    st.dataframe(predictions)
    
    st.write("### Detected Anomalies")
    st.dataframe(anomalies)
    
    if not anomalies.empty:
        st.warning("High risk detected! Immediate action required.")
