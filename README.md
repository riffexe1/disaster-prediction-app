# AI-Based Disaster Prediction & Response System

A student Streamlit project combining data exploration, machine learning, and LLM-generated disaster analysis and response suggestions.

## LLM integration

The main prediction and evacuation workflows use Groq with `llama-3.3-70b-versatile`. The implementation is in `utils/gemini_integration.py`; the filename and `GeminiPredictor` class retain their earlier names.

The prediction prompt supplies a historical-data summary, machine-learning results, and the disaster type. It requests organized sections covering risk, temporal patterns, warning indicators, geographic zones, mitigation, confidence, and trends. The evacuation workflow requests response recommendations.

## Features

- Kaggle dataset download and data preprocessing
- Interactive exploratory charts using Plotly
- Machine-learning training and evaluation
- LLM-generated analysis and evacuation suggestions
- Streamlit dashboard and data exports

## Run locally

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Supply your own Kaggle credentials and Groq API key through the app setup interface. Never commit these credentials. Data and model directories are created by the application.

## Repository contents

`app.py` is the main entry point. `app_updated.py` is an alternative version retained from the original project. `config.py` contains settings and dataset identifiers; `utils/` contains data, visualization, machine-learning, and LLM modules.

This source-only sharing copy excludes the original virtual environment, cached bytecode, datasets, and serialized model. Re-download suitable data and train models locally. Dataset availability and suitability must be verified; a file named `globalterrorismdb_0718dist.csv` was present under the original flood directory and is excluded here.

## Current limitations

This is an educational prototype, not a validated disaster forecast or emergency guidance service. LLM outputs and confidence statements are generated text, not calibrated forecasts. Historical earthquake data does not establish reliable earthquake prediction.

Some auxiliary methods in the LLM module still call the older Gemini-style `generate_content` API and require migration before use. Dependencies are not pinned. This sharing copy passed Python syntax checks and a pattern-based credential scan; the application and external API calls were not run or verified during packaging.
