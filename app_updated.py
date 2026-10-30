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
        st.write(f"Gemini AI: {'✅' if st.session_state.gemini_configured else '❌'}")
        st.write(f"Flood Data: {'✅' if st.session_state.flood_data is not None else '❌'}")
        st.write(f"Earthquake Data: {'✅' if st.session_state.earthquake_data is not None else '❌'}")
        st.write(f"Models Trained: {len(st.session_state.ml_models)}")
    
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
        st.subheader("🔑 Step 2: Configure Gemini AI")
        st.info("Enter your Google Gemini API key (using FREE gemini-2.0-flash-exp model).")
        
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if st.button("Configure Gemini AI", type="primary"):
            if gemini_api_key:
                gemini_predictor = GeminiPredictor(gemini_api_key)
                st.session_state.gemini_configured = True
                st.session_state.gemini_api_key = gemini_api_key
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
            download_path = os.path.join(config.DATA_DIR, 'flood')
            result = kaggle_downloader.download_dataset(selected_flood, download_path, 'flood')
            if result:
                st.session_state.datasets_downloaded['flood'] = result
                # Auto-load flood data
                df = data_processor.load_data(result)
                if df is not None:
                    df = data_processor.clean_data(df)
                    st.session_state.flood_data = df
    
    with col2:
        st.subheader("🌍 Earthquake Data")
        earthquake_datasets = kaggle_downloader.get_available_datasets()['earthquake']
        selected_earthquake = st.selectbox("Select Earthquake Dataset:", earthquake_datasets, key="eq_select")
        
        if st.button("📥 Download Earthquake Data", type="primary", key="download_eq"):
            download_path = os.path.join(config.DATA_DIR, 'earthquake')
            result = kaggle_downloader.download_dataset(selected_earthquake, download_path, 'earthquake')
            if result:
                st.session_state.datasets_downloaded['earthquake'] = result
                # Auto-load earthquake data
                df = data_processor.load_data(result)
                if df is not None:
                    df = data_processor.clean_data(df)
                    st.session_state.earthquake_data = df
    
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
    
    # Select dataset
    available_datasets = []
    if st.session_state.flood_data is not None:
        available_datasets.append("Flood Data")
    if st.session_state.earthquake_data is not None:
        available_datasets.append("Earthquake Data")
    
    selected_dataset = st.selectbox("Select Dataset:", available_datasets, key="ml_dataset")
    df = st.session_state.flood_data if "Flood" in selected_dataset else st.session_state.earthquake_data
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_models = ["Random Forest", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
        
        model_type = st.selectbox("Model Type:", available_models)
    
    with col2:
        task_type = st.selectbox("Task:", ["Regression", "Classification"])
    
    with col3:
        n_estimators = st.slider("N Estimators:", 50, 200, 100)
    
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
    if st.button("🚀 Train Models", type="primary", key="train_ml"):
        for target_col in target_cols:
            st.subheader(f"Training model for: {target_col}")
            
            with st.spinner(f"Training {model_type} for {target_col}..."):
                # Initialize model
                model = DisasterPredictionModel(
                    model_type='classification' if task_type == "Classification" else 'regression'
                )
                
                # Prepare data
                X_train, X_test, y_train, y_test = model.prepare_data(
                    df, target_col, feature_cols
                )
                
                if X_train is not None:
                    # Train model
                    if model_type == "Random Forest":
                        model.train_random_forest(X_train, y_train, n_estimators)
                    elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                        model.train_xgboost(X_train, y_train, n_estimators)
                    else:
                        # Use Gradient Boosting as alternative
                        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                        if task_type == "Classification":
                            model.model = GradientBoostingClassifier(n_estimators=n_estimators)
                        else:
                            model.model = GradientBoostingRegressor(n_estimators=n_estimators)
                        model.model.fit(X_train, y_train)
                        st.success("✅ Gradient Boosting model trained!")
                    
                    # Evaluate
                    metrics = model.evaluate_model(X_test, y_test)
                    
                    # Save model
                    model_key = f"{selected_dataset}_{target_col}_{model_type}"
                    st.session_state.ml_models[model_key] = {
                        'model': model,
                        'metrics': metrics,
                        'target': target_col,
                        'features': feature_cols
                    }
                    
                    # Show feature importance
                    if model.feature_importance:
                        with st.expander(f"Feature Importance for {target_col}"):
                            feature_names = list(model.feature_importance.keys())
                            importances = list(model.feature_importance.values())
                            fig = eda_visualizer.plot_feature_importance(feature_names, importances)
                            st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"✅ All models trained successfully! Total: {len(st.session_state.ml_models)}")
    
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
        st.warning("⚠️ Please configure Gemini AI in the Home & Setup page first!")
        return
    
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
    
    disaster_type = st.selectbox("Select Disaster Type:", available_datasets)
    df = st.session_state.flood_data if disaster_type == "Flood" else st.session_state.earthquake_data
    
    # Tabs for different predictions
    tab1, tab2 = st.tabs(["🔮 Future Predictions", "🚨 Evacuation Plan"])
    
    with tab1:
        st.subheader("Generate Future Disaster Predictions")
        
        if st.button("🤖 Generate AI Predictions", type="primary", key="gen_pred"):
            with st.spinner("Generating predictions with Gemini AI..."):
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
                    st.success("✅ Predictions generated!")
                    st.markdown(predictions['prediction_text'])
                    
                    st.download_button(
                        "📥 Download Predictions",
                        predictions['prediction_text'],
                        f"predictions_{disaster_type}_{datetime.now().strftime('%Y%m%d')}.txt"
                    )
    
    with tab2:
        st.subheader("Evacuation & Response Recommendations")
        
        severity = st.select_slider("Severity:", ["Low", "Medium", "High", "Critical"])
        
        if st.button("🚨 Generate Evacuation Plan", type="primary", key="gen_evac"):
            with st.spinner("Generating recommendations..."):
                recommendations = gemini_predictor.generate_evacuation_recommendations(
                    {'disaster': disaster_type}, disaster_type.lower(), severity.lower()
                )
                
                if recommendations:
                    st.success("✅ Plan generated!")
                    st.markdown(recommendations['recommendations'])
    
    st.divider()
    
    # Next button
    if st.button("➡️ Next: View Results", type="primary", key="next_pred"):
        next_page()


def show_results():
    """Results page"""
    
    st.header("📈 Comprehensive Results & Insights")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        flood_count = len(st.session_state.flood_data) if st.session_state.flood_data is not None else 0
        st.metric("Flood Records", flood_count)
    
    with col2:
        eq_count = len(st.session_state.earthquake_data) if st.session_state.earthquake_data is not None else 0
        st.metric("Earthquake Records", eq_count)
    
    with col3:
        st.metric("Models Trained", len(st.session_state.ml_models))
    
    with col4:
        st.metric("Status", "✅ Complete")
    
    st.divider()
    
    # Download section
    st.subheader("💾 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.flood_data is not None:
            csv = st.session_state.flood_data.to_csv(index=False)
            st.download_button(
                "📥 Flood Data (CSV)",
                csv,
                f"flood_data_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col2:
        if st.session_state.earthquake_data is not None:
            csv = st.session_state.earthquake_data.to_csv(index=False)
            st.download_button(
                "📥 Earthquake Data (CSV)",
                csv,
                f"earthquake_data_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col3:
        if st.session_state.ml_models:
            models_summary = "\n\n".join([f"{k}: {v['metrics']}" for k, v in st.session_state.ml_models.items()])
            st.download_button(
                "📥 Models Summary",
                models_summary,
                f"models_summary_{datetime.now().strftime('%Y%m%d')}.txt"
            )
    
    st.divider()
    
    st.markdown("""
    ### 🎉 Analysis Complete!
    
    Your disaster prediction system has successfully:
    - ✅ Downloaded and processed disaster datasets
    - ✅ Performed comprehensive exploratory data analysis
    - ✅ Trained multiple machine learning models
    - ✅ Generated AI-powered future predictions
    - ✅ Provided actionable insights and recommendations
    
    **🚀 System is ready for continuous monitoring and predictions!**
    """)


if __name__ == "__main__":
    main()
