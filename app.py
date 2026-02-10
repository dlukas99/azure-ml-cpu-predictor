import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

URL = st.secrets["AZURE_URL"]
KEY = st.secrets["AZURE_KEY"]

st.set_page_config(page_title="CPU Base Clock Predictor v3", layout="wide")

st.title("CPU Base Clock Analytics and Prediction")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Project Deep Dive"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.subheader("Predict Processor Base Clock")
    col1, col2 = st.columns(2)
    
    with col1:
        cores = st.number_input("Cores", min_value=1, value=8)
        threads = st.number_input("Threads", min_value=1, value=16)
        boost_clock = st.number_input("Boost Clock (GHz)", min_value=0.1, value=4.40)
    with col2:
        process = st.number_input("Process (nm)", min_value=1, value=7)
        tdp = st.number_input("TDP (W)", min_value=1, value=105)

    if st.button("Calculate Prediction", type="primary", use_container_width=True):
        payload = {"data": [{
            "cores": cores, 
            "threads": threads, 
            "boost_clock": boost_clock, 
            "process": process, 
            "tdp": tdp
        }]}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
        
        try:
            res = requests.post(URL, json=payload, headers=headers)
            output = res.json()
            if isinstance(output, str): output = json.loads(output)
            
            prediction = output['prediction'][0]
            st.metric(label="Predicted Base Clock", value=f"{prediction:.2f} GHz")
        except:
            st.error("API Connection Error. Verify endpoint status in Azure.")

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.subheader("Batch Prediction Instructions")
    st.info("""
    1. Upload a CSV file.
    2. Data is automatically cleaned and filtered to 5 features: Cores, Threads, Boost Clock, Process, TDP.
    3. The optimized Random Forest v3 model performs inference.
    """)
    
    uploaded_file = st.file_uploader("Upload CPU CSV File", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'released' in df.columns:
            df.dropna(subset=['released'], inplace=True)
        df.drop_duplicates(inplace=True)

        cols_to_fix = ['process', 'boost_clock']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())

        model_features = ['cores', 'threads', 'boost_clock', 'process', 'tdp']
        
        if all(c in df.columns for c in model_features):
            final_df = df[model_features].copy()
            st.subheader("Cleaned Input Preview (Top 5 Features)")
            st.dataframe(final_df.head(10), use_container_width=True)
            
            if st.button("Run Batch Inference"):
                payload = {"data": final_df.to_dict(orient="records")}
                headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
                
                try:
                    res = requests.post(URL, json=payload, headers=headers)
                    output = res.json()
                    if isinstance(output, str): output = json.loads(output)
                    
                    df["predicted_base_clock"] = output["prediction"]
                    st.success("Batch processing complete.")
                    st.dataframe(df, use_container_width=True)
                except:
                    st.error("Batch prediction failed.")

# --- TAB 3: PROJECT DEEP DIVE ---
with tab3:
    st.header("Project Architecture and Advanced Insights")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "Optimized Random Forest")
    m2.metric("Accuracy (R2 Score)", "0.90")
    m3.metric("Feature Selection", "SelectKBest (Filter)")

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Model Performance Comparison")
        comp = pd.DataFrame({
            "Algorithm": ["ElasticNet", "Linear Regression", "Decision Tree", "Gradient Boosting", "Random Forest (Opt)"],
            "R2 Score": [0.52, 0.54, 0.78, 0.88, 0.90]
        })
        st.bar_chart(data=comp, x="Algorithm", y="R2 Score")
        st.write("Comparison of 5 algorithms after hyperparameter tuning.")

    with col_b:
        st.subheader("Top 5 Selected Features")
        importance = pd.DataFrame({
            "Feature": ["TDP", "Boost Clock", "Cores", "Threads", "Process"],
            "Impact": [0.48, 0.32, 0.10, 0.07, 0.03]
        })
        st.bar_chart(data=importance, x="Feature", y="Impact")
        st.write("Importance ranking based on SelectKBest statistical scores.")

    st.markdown("---")
    
    st.subheader("Engineering Workflow")
    st.info("""
    - **Exploratory Data Analysis (EDA)**: Profiling data using boxplots and correlation matrices to identify trends and outliers.
    - **Feature Engineering**: Applied **SelectKBest** with `f_regression` to isolate the 5 most predictive variables.
    - **Hyperparameter Optimization**: Used **GridSearchCV** for cross-validated tuning of the Random Forest regressor.
    - **Cloud Deployment**: Deployed as an ACI container (v3) using the Azure ML SDK.
    """)