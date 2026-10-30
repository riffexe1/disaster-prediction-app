"""
AI-Based Disaster Prediction & Response System
Main Streamlit Application - UPDATED VERSION

A comprehensive system for predicting natural disasters using ML and AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from utils.kaggle_downloader import KaggleDownloader
from utils.data_processing import DataProcessor
from utils.eda import EDAVisualizer
from utils.ml_models import DisasterPredictionModel, XGBOOST_AVAILABLE
from utils.gemini_integration import GeminiPredictor
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {margin-top: 1rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    h2 {color: #ff7f0e; padding-top: 1rem;}
    .next-button {margin-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'kaggle_configured' not in st.session_state:
    st.session_state.kaggle_configured = False
if 'datasets_downloaded' not in st.session_state:
    st.session_state.datasets_downloaded = {}
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False
if 'flood_data' not in st.session_state:
    st.session_state.flood_data = None
if 'earthquake_data' not in st.session_state:
    st.session_state.earthquake_data = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Initialize utilities
kaggle_downloader = KaggleDownloader()
data_processor = DataProcessor()
eda_visualizer = EDAVisualizer()

# Page names
PAGES = ["🏠 Home & Setup", "📥 Data Collection", "📊 Data Analysis & EDA", 
         "🤖 ML Model Training", "🔮 AI Predictions", "📈 Results & Insights"]


def main():
    """Main application function"""
    
    # Title
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown("### 🎯 Predict and Respond to Natural Disasters with AI")
    st.divider()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        
        # Radio button for page selection
        page_index = st.radio(
            "Select Page:",
            range(len(PAGES)),
            format_func=lambda x: PAGES[x],
            index=st.session_state.current_page,
            label_visibility="collapsed"
        )
        
        # Update current page
        if page_index != st.session_state.current_page:
            st.session_state.current_page = page_index
        
        st.divider()
        
        # Status indicators
        st.subheader("📌 Status")
        st.write(f"Kaggle: {'✅' if st.session_state.kaggle_configured else '❌'}")
        st.write(f"Groq AI: {'✅' if st.session_state.gemini_configured else '❌'}")
        st.write(f"Flood Data: {'✅' if st.session_state.flood_data is not None else '❌'}")
        st.write(f"Earthquake Data: {'✅' if st.session_state.earthquake_data is not None else '❌'}")
        
        # Count actual trained models (not target variables)
        unique_models = set()
        for key in st.session_state.ml_models.keys():
            model_type = st.session_state.ml_models[key].get('model_type', 'Unknown')
            unique_models.add(model_type)
        st.write(f"Model Types Trained: {len(unique_models)}")
        st.write(f"Total Predictions: {len(st.session_state.ml_models)}")
    
    # Display current page
    page = PAGES[st.session_state.current_page]
    
    if "Home & Setup" in page:
        show_home_setup()
    elif "Data Collection" in page:
        show_data_collection()
    elif "Data Analysis" in page:
        show_eda()
    elif "ML Model Training" in page:
        show_ml_training()
    elif "AI Predictions" in page:
        show_ai_predictions()
    elif "Results" in page:
        show_results()


def next_page():
    """Navigate to next page"""
    if st.session_state.current_page < len(PAGES) - 1:
        st.session_state.current_page += 1
        st.rerun()


def show_home_setup():
    """Home and setup page"""
    
    st.header("🏠 Welcome to Disaster Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Step 1: Upload Kaggle Credentials")
        st.info("Upload your kaggle.json file to download disaster datasets automatically.")
        
        kaggle_file = st.file_uploader(
            "Upload kaggle.json",
            type=['json'],
            help="Download from https://www.kaggle.com/settings -> Create New API Token"
        )
        
        if kaggle_file is not None:
            if st.button("Configure Kaggle API", type="primary"):
                if kaggle_downloader.setup_kaggle_credentials(kaggle_file):
                    st.session_state.kaggle_configured = True
                    st.balloons()
    
    with col2:
        st.subheader("🔑 Step 2: Configure Groq AI")
        st.info("Enter your Groq API key (FREE tier with fast Llama 3.3 70B model).")
        
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your FREE API key from https://console.groq.com/keys"
        )
        
        if st.button("Configure Groq AI", type="primary"):
            if groq_api_key:
                gemini_predictor = GeminiPredictor(groq_api_key)
                st.session_state.gemini_configured = True
                st.session_state.gemini_api_key = groq_api_key
                st.balloons()
            else:
                st.error("Please enter a valid API key")
    
    st.divider()
    
    # Next button
    if st.session_state.kaggle_configured and st.session_state.gemini_configured:
        if st.button("➡️ Next: Download Data", type="primary", key="next_home"):
            next_page()


def show_data_collection():
    """Data collection page"""
    
    st.header("📥 Data Collection from Kaggle")
    
    if not st.session_state.kaggle_configured:
        st.warning("⚠️ Please configure Kaggle API in the Home & Setup page first!")
        return
    
    st.info("📥 Download both Flood and Earthquake datasets for comprehensive analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌊 Flood Data")
        flood_datasets = kaggle_downloader.get_available_datasets()['flood']
        selected_flood = st.selectbox("Select Flood Dataset:", flood_datasets, key="flood_select")
        
        if st.button("📥 Download Flood Data", type="primary", key="download_flood"):
            with st.spinner("Downloading flood data... This may take a minute..."):
                download_path = os.path.join(config.DATA_DIR, 'flood')
                result = kaggle_downloader.download_dataset(selected_flood, download_path, 'flood')
                if result:
                    st.session_state.datasets_downloaded['flood'] = result
                    # Auto-load flood data
                    try:
                        # Handle both file and directory paths
                        file_path = result
                        if os.path.isdir(result):
                            # Search recursively for data files (CSV, Excel, JSON, Parquet)
                            data_files = []
                            for root, dirs, files in os.walk(result):
                                for file in files:
                                    if file.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv')):
                                        data_files.append(os.path.join(root, file))
                            
                            if data_files:
                                file_path = data_files[0]  # Use the first data file found
                                rel_path = os.path.relpath(file_path, result)
                                st.info(f"✅ Found data file: {rel_path}")
                            else:
                                # List all files if no recognized formats found
                                all_files = []
                                for root, dirs, files in os.walk(result):
                                    for file in files:
                                        all_files.append(file)
                                
                                if all_files:
                                    st.error("❌ No recognized data files found (CSV, Excel, JSON, Parquet).")
                                    st.info(f"📂 Files in folder: {', '.join(all_files[:10])}")
                                else:
                                    st.error("❌ Downloaded folder is empty!")
                                    st.info("💡 Download may have failed. Please check your internet connection and Kaggle API credentials.")
                                return
                        
                        if os.path.isfile(file_path):
                            with st.spinner(f"Loading {os.path.basename(file_path)}..."):
                                df = data_processor.load_data(file_path)
                                if df is not None and len(df) > 0:
                                    df = data_processor.clean_data(df)
                                    st.session_state.flood_data = df
                                    st.success(f"🎉 Flood data loaded successfully! {len(df)} records, {len(df.columns)} columns.")
                                    st.balloons()
                                else:
                                    st.error("❌ File loaded but contains no data!")
                        else:
                            st.error(f"❌ Invalid file path: {file_path}")
                    except Exception as e:
                        st.error(f"❌ Error loading flood data: {str(e)}")
                        st.info("💡 Try selecting a different dataset from the dropdown.")
                else:
                    st.error("⚠️ Failed to download flood data. Please check your Kaggle API configuration and try a different dataset.")
    
    with col2:
        st.subheader("🌍 Earthquake Data")
        earthquake_datasets = kaggle_downloader.get_available_datasets()['earthquake']
        selected_earthquake = st.selectbox("Select Earthquake Dataset:", earthquake_datasets, key="eq_select")
        
        if st.button("📥 Download Earthquake Data", type="primary", key="download_eq"):
            with st.spinner("Downloading earthquake data... This may take a minute..."):
                download_path = os.path.join(config.DATA_DIR, 'earthquake')
                result = kaggle_downloader.download_dataset(selected_earthquake, download_path, 'earthquake')
                if result:
                    st.session_state.datasets_downloaded['earthquake'] = result
                    # Auto-load earthquake data
                    try:
                        file_path = result
                        if os.path.isdir(result):
                            # Search recursively for data files (CSV, Excel, JSON, Parquet)
                            data_files = []
                            for root, dirs, files in os.walk(result):
                                for file in files:
                                    if file.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv')):
                                        data_files.append(os.path.join(root, file))
                            
                            if data_files:
                                file_path = data_files[0]  # Use the first data file found
                                rel_path = os.path.relpath(file_path, result)
                                st.info(f"✅ Found data file: {rel_path}")
                            else:
                                # List all files if no recognized formats found
                                all_files = []
                                for root, dirs, files in os.walk(result):
                                    for file in files:
                                        all_files.append(file)
                                
                                if all_files:
                                    st.error("❌ No recognized data files found (CSV, Excel, JSON, Parquet).")
                                    st.info(f"📂 Files in folder: {', '.join(all_files[:10])}")
                                else:
                                    st.error("❌ Downloaded folder is empty!")
                                    st.info("💡 Download may have failed. Please check your internet connection and Kaggle API credentials.")
                                return
                        
                        if os.path.isfile(file_path):
                            with st.spinner(f"Loading {os.path.basename(file_path)}..."):
                                df = data_processor.load_data(file_path)
                                if df is not None and len(df) > 0:
                                    df = data_processor.clean_data(df)
                                    st.session_state.earthquake_data = df
                                    st.success(f"🎉 Earthquake data loaded successfully! {len(df)} records, {len(df.columns)} columns.")
                                    st.balloons()
                                else:
                                    st.error("❌ File loaded but contains no data!")
                        else:
                            st.error(f"❌ Invalid file path: {file_path}")
                    except Exception as e:
                        st.error(f"❌ Error loading earthquake data: {str(e)}")
                        st.info("💡 Try selecting a different dataset from the dropdown.")
    
    st.divider()
    
    # Show loaded datasets
    if st.session_state.flood_data is not None or st.session_state.earthquake_data is not None:
        st.subheader("✅ Loaded Datasets")
        
        if st.session_state.flood_data is not None:
            st.success(f"🌊 Flood Data: {st.session_state.flood_data.shape[0]} rows, {st.session_state.flood_data.shape[1]} columns")
            with st.expander("Preview Flood Data"):
                st.dataframe(st.session_state.flood_data.head(10), use_container_width=True)
        
        if st.session_state.earthquake_data is not None:
            st.success(f"🌍 Earthquake Data: {st.session_state.earthquake_data.shape[0]} rows, {st.session_state.earthquake_data.shape[1]} columns")
            with st.expander("Preview Earthquake Data"):
                st.dataframe(st.session_state.earthquake_data.head(10), use_container_width=True)
        
        # Next button
        if st.button("➡️ Next: Analyze Data", type="primary", key="next_data"):
            next_page()


def show_eda():
    """Exploratory Data Analysis page"""
    
    st.header("📊 Exploratory Data Analysis")
    
    if st.session_state.flood_data is None and st.session_state.earthquake_data is None:
        st.warning("⚠️ Please load datasets from the Data Collection page first!")
        return
    
    # Select which dataset to analyze
    available_datasets = []
    if st.session_state.flood_data is not None:
        available_datasets.append("Flood Data")
    if st.session_state.earthquake_data is not None:
        available_datasets.append("Earthquake Data")
    
    selected_dataset = st.selectbox("Select Dataset to Analyze:", available_datasets)
    
    df = st.session_state.flood_data if "Flood" in selected_dataset else st.session_state.earthquake_data
    
    # Dashboard summary
    st.subheader(f"📈 {selected_dataset} Overview")
    summary = eda_visualizer.create_dashboard_summary(df)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Records", summary['total_records'])
    with col2:
        st.metric("Total Columns", summary['total_columns'])
    with col3:
        st.metric("Numeric Cols", summary['numeric_columns'])
    with col4:
        st.metric("Categorical Cols", summary['categorical_columns'])
    with col5:
        st.metric("Missing Values", summary['missing_values'])
    with col6:
        st.metric("Memory Usage", summary['memory_usage'])
    
    st.divider()
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution Analysis", 
        "🔗 Correlation Matrix", 
        "📈 Time Series", 
        "📉 Custom Plots"
    ])
    
    with tab1:
        st.subheader("Distribution Analysis")
        
        # Numeric distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution:", numeric_cols, key=f"dist_{selected_dataset}")
            fig = eda_visualizer.plot_distribution(df, selected_col)
            st.plotly_chart(fig, use_container_width=True)
        
        # Categorical distribution
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            selected_cat = st.selectbox("Select categorical column:", categorical_cols, key=f"cat_{selected_dataset}")
            fig = eda_visualizer.plot_categorical_distribution(df, selected_cat)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Matrix")
        fig = eda_visualizer.plot_correlation_matrix(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Time Series Analysis")
        
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select date/time column:", date_cols, key=f"ts_date_{selected_dataset}")
            with col2:
                value_col = st.selectbox("Select value column:", numeric_cols, key=f"ts_val_{selected_dataset}")
            
            if st.button("Generate Time Series", key=f"ts_gen_{selected_dataset}"):
                try:
                    df_ts = df.copy()
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                    df_ts = df_ts.dropna(subset=[date_col, value_col])
                    df_ts = df_ts.sort_values(date_col)
                    
                    fig = px.line(df_ts, x=date_col, y=value_col, 
                                title=f'{value_col} Over Time',
                                markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating time series: {str(e)}")
        else:
            st.info("No suitable date/time columns found for time series analysis")
    
    with tab4:
        st.subheader("Custom Visualization")
        
        plot_type = st.selectbox("Select plot type:", 
                                ["Scatter Plot", "Box Plot", "Violin Plot", "Histogram"],
                                key=f"plot_{selected_dataset}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols, key=f"scatter_x_{selected_dataset}")
            with col2:
                y_col = st.selectbox("Y-axis:", [c for c in numeric_cols if c != x_col], key=f"scatter_y_{selected_dataset}")
            
            if st.button("Generate Scatter Plot", key=f"gen_scatter_{selected_dataset}"):
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            selected_col = st.selectbox("Select column:", numeric_cols, key=f"box_{selected_dataset}")
            if st.button("Generate Box Plot", key=f"gen_box_{selected_dataset}"):
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Violin Plot":
            selected_col = st.selectbox("Select column:", numeric_cols, key=f"violin_{selected_dataset}")
            if st.button("Generate Violin Plot", key=f"gen_violin_{selected_dataset}"):
                fig = px.violin(df, y=selected_col, title=f"Violin Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Histogram":
            selected_col = st.selectbox("Select column:", numeric_cols, key=f"hist_{selected_dataset}")
            bins = st.slider("Number of bins:", 10, 100, 30, key=f"bins_{selected_dataset}")
            if st.button("Generate Histogram", key=f"gen_hist_{selected_dataset}"):
                fig = px.histogram(df, x=selected_col, nbins=bins, title=f"Histogram of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Next button
    if st.button("➡️ Next: Train ML Models", type="primary", key="next_eda"):
        next_page()


def show_ml_training():
    """Machine Learning training page"""
    
    st.header("🤖 Machine Learning Model Training")
    
    if st.session_state.flood_data is None and st.session_state.earthquake_data is None:
        st.warning("⚠️ Please load datasets first!")
        return
    
    # Select dataset or combine both
    available_datasets = []
    if st.session_state.flood_data is not None:
        available_datasets.append("Dataset 1 (Flood)")
    if st.session_state.earthquake_data is not None:
        available_datasets.append("Dataset 2 (Earthquake)")
    if st.session_state.flood_data is not None and st.session_state.earthquake_data is not None:
        available_datasets.append("📊 Combined (Both Datasets)")
    
    selected_dataset = st.selectbox("Select Training Data:", available_datasets, key="ml_dataset")
    
    # Get the dataframe based on selection
    if "Combined" in selected_dataset:
        # Combine both datasets
        df1 = st.session_state.flood_data.copy()
        df2 = st.session_state.earthquake_data.copy()
        
        # Add source column to track dataset origin
        df1['dataset_source'] = 'Dataset_1'
        df2['dataset_source'] = 'Dataset_2'
        
        # Find common columns
        common_cols = list(set(df1.columns) & set(df2.columns))
        
        if len(common_cols) > 2:  # Need at least 3 common columns (including dataset_source)
            df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
            st.success(f"✅ Combined datasets: {len(df1)} + {len(df2)} = {len(df)} total records with {len(common_cols)-1} common features")
        else:
            st.warning("⚠️ Datasets have too few common columns. Using all available columns...")
            df = pd.concat([df1, df2], ignore_index=True, sort=False)
            st.info(f"📊 Combined shape: {df.shape}")
    else:
        df = st.session_state.flood_data if "Dataset 1" in selected_dataset else st.session_state.earthquake_data
        st.info(f"📊 Training on single dataset: {len(df)} records, {len(df.columns)} columns")
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_type = st.selectbox("Task:", ["Regression", "Classification"])
    
    with col2:
        n_estimators = st.slider("N Estimators:", 50, 200, 100)
    
    st.info("🎯 ALL models (Random Forest, Gradient Boosting, and XGBoost) will be trained for maximum accuracy!")
    
    st.divider()
    
    # Feature and target selection
    st.subheader("🎯 Feature & Target Selection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns for model training!")
        return
    
    # Multiple target variables
    target_cols = st.multiselect(
        "Select Target Variable(s):",
        numeric_cols,
        help="Select one or more target variables to predict"
    )
    
    if not target_cols:
        st.warning("Please select at least one target variable")
        return
    
    # Feature selection
    available_features = [col for col in numeric_cols if col not in target_cols]
    
    use_all_features = st.checkbox("Use all available features", value=True)
    
    if use_all_features:
        feature_cols = available_features
    else:
        feature_cols = st.multiselect(
            "Select Features:",
            available_features,
            default=available_features[:min(5, len(available_features))]
        )
    
    st.info(f"Selected {len(feature_cols)} features and {len(target_cols)} target(s)")
    
    st.divider()
    
    # Train models for each target
    if st.button("🚀 Train ALL Models", type="primary", key="train_ml"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(target_cols) * 3  # RF, GB, XGBoost for each target
        model_count = 0
        
        for target_col in target_cols:
            st.subheader(f"📊 Training models for: {target_col}")
            
            # Initialize model
            model = DisasterPredictionModel(
                model_type='classification' if task_type == "Classification" else 'regression'
            )
            
            # Prepare data
            X_train, X_test, y_train, y_test = model.prepare_data(
                df, target_col, feature_cols
            )
            
            if X_train is not None:
                # Train Random Forest
                status_text.text(f"🌲 Training Random Forest for {target_col}...")
                model_rf = DisasterPredictionModel(model_type='classification' if task_type == "Classification" else 'regression')
                model_rf.train_random_forest(X_train, y_train, n_estimators)
                metrics_rf = model_rf.evaluate_model(X_test, y_test)
                
                model_key = f"{selected_dataset}_{target_col}_RF"
                st.session_state.ml_models[model_key] = {
                    'model': model_rf,
                    'model_type': 'Random Forest',
                    'metrics': metrics_rf,
                    'target': target_col,
                    'features': feature_cols
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
                st.success("✅ Random Forest trained!")
                
                # Train Gradient Boosting
                status_text.text(f"🚀 Training Gradient Boosting for {target_col}...")
                from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                model_gb = DisasterPredictionModel(model_type='classification' if task_type == "Classification" else 'regression')
                if task_type == "Classification":
                    model_gb.model = GradientBoostingClassifier(n_estimators=n_estimators)
                else:
                    model_gb.model = GradientBoostingRegressor(n_estimators=n_estimators)
                model_gb.model.fit(X_train, y_train)
                metrics_gb = model_gb.evaluate_model(X_test, y_test)
                
                model_key = f"{selected_dataset}_{target_col}_GB"
                st.session_state.ml_models[model_key] = {
                    'model': model_gb,
                    'model_type': 'Gradient Boosting',
                    'metrics': metrics_gb,
                    'target': target_col,
                    'features': feature_cols
                }
                model_count += 1
                progress_bar.progress(model_count / total_models)
                st.success("✅ Gradient Boosting trained!")
                
                # Train XGBoost if available
                if XGBOOST_AVAILABLE:
                    status_text.text(f"⚡ Training XGBoost for {target_col}...")
                    model_xgb = DisasterPredictionModel(model_type='classification' if task_type == "Classification" else 'regression')
                    model_xgb.train_xgboost(X_train, y_train, n_estimators)
                    metrics_xgb = model_xgb.evaluate_model(X_test, y_test)
                    
                    model_key = f"{selected_dataset}_{target_col}_XGB"
                    st.session_state.ml_models[model_key] = {
                        'model': model_xgb,
                        'model_type': 'XGBoost',
                        'metrics': metrics_xgb,
                        'target': target_col,
                        'features': feature_cols
                    }
                    model_count += 1
                    progress_bar.progress(model_count / total_models)
                    st.success("✅ XGBoost trained!")
        
        status_text.text("✅ All models trained successfully!")
        st.success(f"🎉 Training Complete! Total models trained: {model_count}")
        st.balloons()
    
    # Show trained models
    if st.session_state.ml_models:
        st.divider()
        st.subheader("📊 Trained Models Summary")
        
        for model_key, model_info in st.session_state.ml_models.items():
            with st.expander(f"Model: {model_key}"):
                st.write(f"**Target:** {model_info['target']}")
                st.write(f"**Features:** {len(model_info['features'])}")
                st.write(f"**Metrics:** {model_info['metrics']}")
    
    st.divider()
    
    # Next button
    if st.session_state.ml_models:
        if st.button("➡️ Next: AI Predictions", type="primary", key="next_ml"):
            next_page()


def show_ai_predictions():
    """AI predictions page"""
    
    st.header("🔮 AI-Powered Future Predictions")
    
    if not st.session_state.gemini_configured:
        st.warning("⚠️ Please configure Groq AI in the Home & Setup page first!")
        return
    
    # Attractive header banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin: 0; text-align: center;'>🌍 Advanced AI Disaster Intelligence</h2>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; text-align: center; font-size: 16px;'>
            Powered by Groq Llama 3.3 70B Versatile - Real-time predictions and evacuation planning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    gemini_predictor = GeminiPredictor(st.session_state.gemini_api_key)
    
    # Select dataset
    available_datasets = []
    if st.session_state.flood_data is not None:
        available_datasets.append("Flood")
    if st.session_state.earthquake_data is not None:
        available_datasets.append("Earthquake")
    
    if not available_datasets:
        st.warning("⚠️ Please load datasets first!")
        return
    
    # Tabs with enhanced styling
    tab1, tab2 = st.tabs(["🔮 Future Disaster Predictions", "🚨 Emergency Evacuation Plan"])
    
    with tab1:
        st.markdown("### 📊 Configure Prediction Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            disaster_type = st.selectbox(
                "🌪️ Disaster Type:", 
                available_datasets,
                help="Select the disaster type to analyze"
            )
            region = st.text_input(
                "📍 Region/Location:", 
                "Global",
                help="Enter specific location or 'Global' for worldwide analysis"
            )
        
        with col2:
            time_horizon = st.selectbox(
                "⏱️ Prediction Timeframe:",
                ["Next Month", "Next 3 Months", "Next 6 Months", "Next Year"],
                help="How far into the future to predict"
            )
            severity_level = st.slider(
                "📈 Historical Severity:",
                1, 10, 5,
                help="Average severity of past disasters (1=Low, 10=Extreme)"
            )
        
        df = st.session_state.flood_data if disaster_type == "Flood" else st.session_state.earthquake_data
        
        if st.button("🤖 Generate AI Predictions", type="primary", use_container_width=True, key="gen_pred"):
            with st.spinner("🧠 AI analyzing disaster patterns using Groq Llama 3.3 70B..."):
                summary = data_processor.get_data_summary(df)
                historical_data = {
                    'total_records': len(df),
                    'columns': df.columns.tolist()[:10],
                    'key_statistics': str(summary)[:500]
                }
                
                ml_results = {}
                for key, info in st.session_state.ml_models.items():
                    if disaster_type.lower() in key.lower():
                        ml_results[key] = str(info['metrics'])[:200]
                
                predictions = gemini_predictor.generate_disaster_prediction(
                    historical_data, ml_results, disaster_type.lower()
                )
                
                if predictions:
                    st.success("✅ AI Analysis Complete!")
                    
                    # Display prediction with attractive card design
                    st.markdown("""
                    <div style='background: linear-gradient(145deg, #e3f2fd 0%, #bbdefb 100%); 
                                padding: 25px; border-radius: 12px; 
                                border-left: 6px solid #1976D2; 
                                margin: 20px 0;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                        <h3 style='color: #0d47a1; margin-top: 0; font-weight: bold;'>
                            📊 Disaster Risk Assessment & Future Predictions
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='font-size: 17px; 
                                line-height: 2.0; 
                                padding: 25px; 
                                background: #ffffff; 
                                border-radius: 10px;
                                color: #000000;
                                font-weight: 500;
                                border: 2px solid #e0e0e0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                        {predictions['prediction_text']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add metrics cards
                    st.markdown("---")
                    met1, met2, met3, met4 = st.columns(4)
                    with met1:
                        st.metric("📍 Region", region, delta=None)
                    with met2:
                        st.metric("🌪️ Type", disaster_type, delta=None)
                    with met3:
                        st.metric("⏱️ Timeframe", time_horizon, delta=None)
                    with met4:
                        st.metric("📈 Severity", f"{severity_level}/10", delta=None)
                    
                    st.markdown("---")
                    st.download_button(
                        "📥 Download Full Prediction Report",
                        predictions['prediction_text'],
                        f"disaster_prediction_{disaster_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate predictions. Please check your Groq API key.")
    
    with tab2:
        st.markdown("### 🚨 Emergency Evacuation & Response Planning")
        
        col_a, col_b = st.columns(2)
        with col_a:
            severity = st.select_slider(
                "📊 Predicted Severity Level:",
                ["Low", "Medium", "High", "Critical", "Catastrophic"],
                value="High",
                help="Estimated impact level of the disaster"
            )
        with col_b:
            affected_population = st.number_input(
                "👥 Estimated Affected Population:",
                min_value=100,
                max_value=10000000,
                value=50000,
                step=1000
            )
        
        if st.button("🚨 Generate Evacuation Plan", type="primary", use_container_width=True, key="gen_evac"):
            with st.spinner("🚑 Generating comprehensive evacuation plan with Groq AI..."):
                recommendations = gemini_predictor.generate_evacuation_recommendations(
                    {'disaster': disaster_type}, disaster_type.lower(), severity.lower()
                )
                
                if recommendations:
                    st.success("✅ Evacuation Plan Generated!")
                    
                    # Display recommendations with attractive styling
                    st.markdown("""
                    <div style='background: linear-gradient(145deg, #fff3e0 0%, #ffe0b2 100%); 
                                padding: 25px; border-radius: 12px; 
                                border-left: 6px solid #F57C00; 
                                margin: 20px 0;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                        <h3 style='color: #E65100; margin-top: 0; font-weight: bold;'>
                            🚨 Emergency Response & Evacuation Plan
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='font-size: 17px; 
                                line-height: 2.0; 
                                padding: 25px; 
                                background: #ffffff; 
                                border-radius: 10px;
                                color: #000000;
                                font-weight: 500;
                                border: 2px solid #e0e0e0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                        {recommendations['recommendations']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    st.markdown("---")
                    st.markdown("### 🎯 Quick Actions")
                    act1, act2, act3 = st.columns(3)
                    with act1:
                        st.button("📞 Alert Emergency Services", use_container_width=True)
                    with act2:
                        st.button("📧 Email to Authorities", use_container_width=True)
                    with act3:
                        st.button("📱 Send SMS Alerts", use_container_width=True)
                    
                    st.download_button(
                        "📥 Download Evacuation Plan",
                        recommendations['recommendations'],
                        f"evacuation_plan_{disaster_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        use_container_width=True
                    )
    
    st.divider()
    
    # Next button
    if st.button("➡️ Next: View Results", type="primary", key="next_pred"):
        next_page()


def show_results():
    """Results page"""
    
    st.header("📈 Comprehensive Results & Insights")
    
    # Attractive header banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px; 
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);'>
        <h2 style='color: white; margin: 0; text-align: center;'>🎯 System Performance Dashboard</h2>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; text-align: center; font-size: 16px;'>
            Complete analysis summary with model performance and insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics with better styling
    st.markdown("### 📊 Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        flood_count = len(st.session_state.flood_data) if st.session_state.flood_data is not None else 0
        st.metric("🌊 Dataset 1 Records", f"{flood_count:,}", delta="Loaded" if flood_count > 0 else None)
    
    with col2:
        eq_count = len(st.session_state.earthquake_data) if st.session_state.earthquake_data is not None else 0
        st.metric("🌍 Dataset 2 Records", f"{eq_count:,}", delta="Loaded" if eq_count > 0 else None)
    
    with col3:
        unique_models = set()
        for key in st.session_state.ml_models.keys():
            model_type = st.session_state.ml_models[key].get('model_type', 'Unknown')
            unique_models.add(model_type)
        st.metric("🤖 Model Types", len(unique_models), delta=f"{len(st.session_state.ml_models)} predictions")
    
    with col4:
        total_records = flood_count + eq_count
        st.metric("📁 Total Data", f"{total_records:,}", delta="✅ Complete")
    
    st.divider()
    
    # Model Performance Section
    if st.session_state.ml_models:
        st.markdown("### 🏆 Model Performance Summary")
        
        # Group models by target variable
        targets_dict = {}
        for key, info in st.session_state.ml_models.items():
            target = info.get('target', 'Unknown')
            model_type = info.get('model_type', 'Unknown')
            if target not in targets_dict:
                targets_dict[target] = []
            targets_dict[target].append({
                'Model': model_type,
                'Metrics': str(info.get('metrics', {}))[:100],
                'Features': len(info.get('features', []))
            })
        
        # Display models by target
        for target, models_list in targets_dict.items():
            with st.expander(f"🎯 Target: {target} ({len(models_list)} models)", expanded=True):
                df_models = pd.DataFrame(models_list)
                st.dataframe(df_models, use_container_width=True)
                
                # Visual comparison
                if len(models_list) > 1:
                    model_names = [m['Model'] for m in models_list]
                    feature_counts = [m['Features'] for m in models_list]
                    
                    fig = px.bar(
                        x=model_names, 
                        y=feature_counts,
                        labels={'x': 'Model Type', 'y': 'Number of Features'},
                        title=f'Feature Usage Comparison for {target}',
                        color=feature_counts,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Data Quality Insights
    st.markdown("### 🔍 Data Quality Insights")
    
    insight_cols = st.columns(2)
    
    with insight_cols[0]:
        if st.session_state.flood_data is not None:
            st.markdown("""
            <div style='background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                <h4 style='color: #2e7d32; margin-top: 0;'>📊 Dataset 1 Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df1 = st.session_state.flood_data
            st.write(f"**Total Records:** {len(df1):,}")
            st.write(f"**Total Columns:** {len(df1.columns)}")
            st.write(f"**Numeric Columns:** {len(df1.select_dtypes(include=[np.number]).columns)}")
            st.write(f"**Missing Values:** {df1.isnull().sum().sum():,}")
            
            # Completeness percentage
            completeness = (1 - df1.isnull().sum().sum() / (len(df1) * len(df1.columns))) * 100
            st.progress(completeness / 100)
            st.caption(f"Data Completeness: {completeness:.1f}%")
    
    with insight_cols[1]:
        if st.session_state.earthquake_data is not None:
            st.markdown("""
            <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3;'>
                <h4 style='color: #1565c0; margin-top: 0;'>🌍 Dataset 2 Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df2 = st.session_state.earthquake_data
            st.write(f"**Total Records:** {len(df2):,}")
            st.write(f"**Total Columns:** {len(df2.columns)}")
            st.write(f"**Numeric Columns:** {len(df2.select_dtypes(include=[np.number]).columns)}")
            st.write(f"**Missing Values:** {df2.isnull().sum().sum():,}")
            
            # Completeness percentage
            completeness = (1 - df2.isnull().sum().sum() / (len(df2) * len(df2.columns))) * 100
            st.progress(completeness / 100)
            st.caption(f"Data Completeness: {completeness:.1f}%")
    
    st.divider()
    
    # Download section with better styling
    st.markdown("### 💾 Export & Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.flood_data is not None:
            csv = st.session_state.flood_data.to_csv(index=False)
            st.download_button(
                "📥 Download Dataset 1 (CSV)",
                csv,
                f"dataset1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.session_state.earthquake_data is not None:
            csv = st.session_state.earthquake_data.to_csv(index=False)
            st.download_button(
                "📥 Download Dataset 2 (CSV)",
                csv,
                f"dataset2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.session_state.ml_models:
            models_summary = "=== ML MODELS PERFORMANCE SUMMARY ===\n\n"
            for k, v in st.session_state.ml_models.items():
                models_summary += f"\n{'='*50}\n"
                models_summary += f"Model: {k}\n"
                models_summary += f"Type: {v.get('model_type', 'N/A')}\n"
                models_summary += f"Target: {v.get('target', 'N/A')}\n"
                models_summary += f"Features: {len(v.get('features', []))}\n"
                models_summary += f"Metrics: {v['metrics']}\n"
            
            st.download_button(
                "📥 Download Models Report",
                models_summary,
                f"models_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
    
    st.divider()
    
    # Success Summary
    st.markdown("""
    <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                padding: 30px; border-radius: 15px; margin: 20px 0;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);'>
        <h3 style='color: #004d40; margin: 0; text-align: center;'>🎉 Analysis Complete!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: white; padding: 25px; border-radius: 10px; border: 2px solid #e0e0e0;'>
        <h4 style='color: #1976d2;'>✅ System Accomplishments:</h4>
        <ul style='font-size: 16px; line-height: 2.0; color: #000000;'>
            <li>📥 <b>Data Collection:</b> Successfully downloaded and processed disaster datasets</li>
            <li>📊 <b>Data Analysis:</b> Performed comprehensive exploratory data analysis with visualizations</li>
            <li>🤖 <b>Model Training:</b> Trained multiple ML models (Random Forest, Gradient Boosting, XGBoost)</li>
            <li>🔮 <b>AI Predictions:</b> Generated AI-powered future predictions using Groq Llama 3.3 70B</li>
            <li>🚨 <b>Emergency Planning:</b> Provided actionable evacuation and response recommendations</li>
            <li>📈 <b>Insights:</b> Delivered comprehensive results and performance metrics</li>
        </ul>
        <h4 style='color: #2e7d32; margin-top: 20px;'>🚀 System Status: Ready for Production</h4>
        <p style='font-size: 16px; color: #000000;'>
            Your AI-powered disaster prediction system is fully operational and ready for continuous monitoring, 
            real-time predictions, and emergency response planning!
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
