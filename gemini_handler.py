import google.generativeai as genai
import streamlit as st # Import Streamlit to access st.secrets
import pandas as pd
import os # Keep os for potential fallback or other environment variables

# --- Configure Gemini API ---
api_key_configured = False
try:
    # Prioritize Streamlit secrets
    if "gemini" in st.secrets and "api_key" in st.secrets.gemini:
        api_key = st.secrets.gemini.api_key
        if api_key: # Ensure the key from secrets is not empty
            genai.configure(api_key=api_key)
            api_key_configured = True
            print("Gemini API Key configured using Streamlit secrets.") # For server logs
        else:
            print("Warning: Gemini API Key found in Streamlit secrets but it is empty.")
    else:
        # Fallback to environment variable if not in Streamlit secrets (optional, can be removed if only secrets are desired)
        print("Gemini API Key not found in Streamlit secrets. Trying environment variable GOOGLE_API_KEY.")
        env_api_key = os.getenv("GOOGLE_API_KEY")
        if env_api_key:
            genai.configure(api_key=env_api_key)
            api_key_configured = True
            print("Gemini API Key configured using GOOGLE_API_KEY environment variable.")
        else:
            print("Warning: GOOGLE_API_KEY environment variable not found or empty.")

    if not api_key_configured:
        st.warning("⚠️ Gemini API Key not configured. Please add it to .streamlit/secrets.toml or set the GOOGLE_API_KEY environment variable. Gemini features will be unavailable.")
        print("Critical Warning: Gemini API Key is not configured from any source.")


except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Gemini features may not work.")
    print(f"Error during Gemini API configuration: {e}")
    api_key_configured = False


# Function to generate response from the model with DataFrame context
def get_gemini_response_with_context(query: str, df: pd.DataFrame):
    if not api_key_configured:
        return "🤖 Gemini API is not configured. Please check your API key setup in .streamlit/secrets.toml."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-flash' etc.

        context = "You are a helpful data analytics assistant. The user has uploaded a dataset."
        if df is not None and not df.empty:
            context += f"\nThe dataset has {df.shape[0]} rows and {df.shape[1]} columns."
            context += f"\nColumn names are: {', '.join(df.columns.tolist())}."
            context += "\nHere's a small sample of the data (first 3 rows):\n"
            context += df.head(3).to_markdown(index=False)
            context += "\nAnd here are the data types:\n"
            context += df.dtypes.to_string()
        else:
            context += "\nNo data is currently loaded or the data is empty."

        full_prompt = f"{context}\n\nUser query: \"{query}\"\n\nPlease provide a concise and helpful answer related to the data if possible, or answer the general query."

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            full_prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                # temperature=0.7,
                # max_output_tokens=1024
            )
        )

        if response.parts:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
             return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            block_reason = ""
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = f" Blocked due to: {response.prompt_feedback.block_reason.name}."
            
            finish_reason = ""
            if response.candidates and response.candidates[0].finish_reason:
                 finish_reason_val = response.candidates[0].finish_reason
                 if hasattr(finish_reason_val, 'name'):
                     finish_reason_val = finish_reason_val.name
                 if finish_reason_val not in ["STOP", "MAX_TOKENS"]:
                    finish_reason = f" Generation finished due to: {finish_reason_val}."
            
            return f"🤖 Gemini couldn't provide an answer.{block_reason}{finish_reason} Please try rephrasing your question or check the safety settings if this persists."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_message = f"🤖 An error occurred while contacting Gemini: {str(e)}"
        if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e) or "Unauthenticated" in str(e):
            error_message = "Error: Gemini API key is not valid, missing, or permission was denied. Please check your .streamlit/secrets.toml."
        
        # Add a button in the UI to show details, or log them for debugging.
        # For now, returning a detailed error string.
        return f"{error_message}\n<details><summary>Click for technical details</summary>\n\n\n{error_details}\n\n</details>"
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import io
import re

# Maximum number of unique values a categorical feature can have to be considered for o… import re
import pandas as pd
from utils import (
    generate_data_summary,
    get_column_stats,
    generate_plot_interactive,
    train_and_evaluate_model,
    get_correlation_matrix
)
from gemini_handler import get_gemini_response_with_context

# Master handler
def handle_query(user_input, df):
    user_input_lower = user_input.lower()

    # --- EDA Commands ---
    if re.search(r"(show|view|display).*(head|first few rows|first \d+ rows)", user_input_lower):
        num_rows = 5
        match = re.search(r"first (\d+) rows", user_input_lower)
        if match:
            num_rows = int(match.group(1))
        return df.head(num_rows) # Returns DataFrame

    if re.search(r"(summary|describe|basic info|overview)", user_input_lower):
        return generate_data_summary(df) # Returns string

    if re.search(r"(column names|show columns|list columns)", user_input_lower):
        return f"*Column Names:*\n\n\n{', '.join(df.columns)}\n"

    if re.search(r"(data types|dtypes|schema)", user_input_lower):
        dtypes_str = df.dtypes.to_string()
        return f"*Data Types:*\n\n\n{dtypes_str}\n"

    if re.search(r"(shape|dimensions|size of data)", user_input_lower):
        return f"*Dataset Shape:*\n\nRows: {df.shape[0]}, Columns: {df.shape[1]}"

    if re.search(r"(missing values|null values|na values|check missing)", user_input_lower):
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if missing_summary.empty:
            return "✅ No missing values found in the dataset."
        return f"*Missing Values per Column:*\n\n\n{missing_summary.to_string()}\n"

    if re.search(r"(correlation|correlation matrix|corr matrix)", user_input_lower):
        # This will return a Plotly figure dictionary
        return get_correlation_matrix(df)


    # --- Stats Commands ---
    if re.search(r"(mean|average|median|mode|std|min|max|stats for).*column", user_input_lower) or \
       re.search(r"stats for ('.'|\".\"|[a-zA-Z0-9_]+)", user_input_lower): # handles "stats for 'column_name'"
        return get_column_stats(user_input_lower, df) # Returns string

    # --- Plot Commands ---
    if re.search(r"(plot|graph|visualize|chart)", user_input_lower):
        # generate_plot_interactive will return a dict for plotly or a string message
        plot_response = generate_plot_interactive(user_input_lower, df)
        return plot_response

    # --- ML Commands ---
    if re.search(r"(train|build|fit|create|develop).*(model|predict|regression|classification)", user_input_lower):
        return train_and_evaluate_model(user_input_lower, df) # Returns string

    # --- Fallback to Gemini ---
    try:
        response = get_gemini_response_with_context(user_input, df)
        return response # Returns string
    except Exception as e:
        return f"🤖 Gemini Error: {str(e)}. Please ensure your API key is configured."