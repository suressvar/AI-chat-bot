import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- UI CONFIGURATION (Bright & Royal Theme) ---
st.set_page_config(
    page_title="Quant AI | Financial Strategist",
    page_icon="💎",
    layout="wide"
)

# Custom CSS for Gemini-style aesthetics
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1a73e8; font-family: 'Google Sans', sans-serif; font-weight: 700; }
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; border: 1px solid #e0e0e0; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    /* AI Message Bubble */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #f0f4f8;
        border: 1px solid #d1e3fa;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. SECURE API CONFIG ---
# This looks for the key in Streamlit's "Secrets" settings instead of hardcoding it
try:
    API_KEY = st.secrets["AIzaSyBaf9O-2zcdIeknGZElbOzTlEU_2Jrb_oA"]
    genai.configure(api_key=AIzaSyBaf9O-2zcdIeknGZElbOzTlEU_2Jrb_oA...)
except Exception:
    st.error("Missing API Key! Please add 'GEMINI_API_KEY' to your Streamlit Secrets.")
    st.stop()

# --- 2. DATA ENGINE ---
@st.cache_data
def process_market_data(file_path):
    try:
        # Load and Clean
        df = pd.read_csv(file_path)
        df = df[['Close/Last']]
        df.columns = ['Close']
        df['Close'] = df['Close'].astype(str).replace({r'[$,]': ''}, regex=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        
        # Chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        df['Day'] = np.arange(len(df))
        
        # Regression Modeling
        X = df[['Day']]
        y = df['Close']
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Future Forecast
        last_day = df['Day'].iloc[-1]
        future_X = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
        future_predictions = model.predict(poly.transform(future_X))
        
        return df, X, y, y_pred, future_X, future_predictions, mse, r2
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

# Load from the local directory (for GitHub/Streamlit deployment)
data_results = process_market_data("Datastock.txt")

if data_results:
    df, X, y, y_pred, future_X, future_preds, mse, r2 = data_results
    
    # --- 3. DASHBOARD UI ---
    st.title("💎 Quant AI Financial Terminal")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Latest Price", f"${y.iloc[-1]:.2f}")
    with m_col2:
        st.metric("30D Target", f"${future_preds[-1]:.2f}", delta=f"{future_preds[-1]-y.iloc[-1]:.2f}")
    with m_col3:
        st.metric("Model Reliability (R²)", f"{r2:.2%}")
    with m_col4:
        st.metric("Volatility (MSE)", f"{mse:.2f}")

    st.divider()

    chart_col, chat_col = st.columns([1.5, 1])

    with chart_col:
        st.subheader("Market Trajectory Analysis")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        ax.plot(X, y, color='#1a73e8', label='Historical Actuals', linewidth=2)
        ax.plot(X, y_pred, color='#dadce0', linestyle='--', label='Model Base')
        ax.plot(future_X, future_preds, color='#d93025', linewidth=3, label='Predictive Projection')
        
        ax.set_facecolor('#ffffff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.legend(frameon=False)
        st.pyplot(fig)

    # --- 4. THE CLEVER AI AGENT ---
    with chat_col:
        st.subheader("Strategist Intelligence")
        
        # System instructions to maximize "Cleverness"
        instruction = f"""
        You are a Tier-1 Quantitative Financial Analyst. 
        Analyze the stock data provided. Look for momentum shifts, pattern acceleration, and risks.
        
        CONTEXT:
        - Current Price: ${y.iloc[-1]:.2f}
        - 30-Day Forecast: ${future_preds[-1]:.2f}
        - Model Confidence (R2): {r2:.4f}
        - Pattern: Polynomial Degree 3
        
        RECENT HISTORY (Last 10 Days):
        {df.tail(10).to_string(index=False)}
        
        GOAL:
        Explain patterns (Bullish/Bearish/Mean Reversion) with high intelligence. 
        Identify if the current curve is overextending or stabilizing.
        Always conclude with a professional disclaimer.
        """

        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Auto-generate initial analysis
            model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=instruction)
            st.session_state.chat = model.start_chat(history=[])
            initial_msg = st.session_state.chat.send_message("Generate a sharp, 3-sentence summary of the current market trajectory and the reliability of our prediction.")
            st.session_state.messages.append({"role": "assistant", "content": initial_msg.text})

        # Chat interface
        chat_container = st.container(height=400)
        for m in st.session_state.messages:
            chat_container.chat_message(m["role"]).markdown(m["content"])

        if prompt := st.chat_input("Inquire about specific patterns or risks..."):
            chat_container.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            response = st.session_state.chat.send_message(prompt)
            chat_container.chat_message("assistant").markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})