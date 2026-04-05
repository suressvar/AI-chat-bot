import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import google.generativeai as genai
import io
import time

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="StockMind AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS  — Dark Gemini-inspired theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;700&family=Google+Sans+Mono&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0d0d0f;
    --surface:     #1a1a1f;
    --surface2:    #22222a;
    --border:      #2e2e3a;
    --accent:      #8ab4f8;
    --accent2:     #c58af9;
    --green:       #81c995;
    --red:         #f28b82;
    --yellow:      #fdd663;
    --text:        #e8eaed;
    --subtext:     #9aa0a6;
    --radius:      16px;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Google Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4285f4, #8ab4f8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: 'Google Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.8rem !important;
    transition: all .2s ease !important;
    letter-spacing: .3px !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(66,133,244,.4) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.4rem !important;
}
[data-testid="stMetricLabel"] { color: var(--subtext) !important; font-size: .85rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.5rem !important; }
[data-testid="stMetricDelta"] { font-size: .8rem !important; }

/* ── Chat / AI bubble ── */
.ai-bubble {
    background: linear-gradient(135deg, #1e2a3a, #1a1f2e);
    border: 1px solid #2d3a52;
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    line-height: 1.75;
    font-size: .97rem;
    color: var(--text);
    position: relative;
    margin-top: .5rem;
}
.ai-bubble::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4285f4, #c58af9, #81c995);
    border-radius: var(--radius) var(--radius) 0 0;
}
.gemini-icon {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: .6rem;
    font-size: .95rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--subtext);
    letter-spacing: .8px;
    text-transform: uppercase;
    margin: 1.2rem 0 .8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: .5rem;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #0d0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(66,133,244,.15), transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #8ab4f8, #c58af9);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}
.hero p { color: var(--subtext); margin: .4rem 0 0; font-size: 1rem; }

/* ── Tag pills ── */
.pill {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: .15rem .75rem;
    font-size: .78rem;
    color: var(--subtext);
    margin: .1rem;
}
.pill.up   { border-color: var(--green); color: var(--green); background: rgba(129,201,149,.08); }
.pill.down { border-color: var(--red);   color: var(--red);   background: rgba(242,139,130,.08); }
.pill.blue { border-color: var(--accent); color: var(--accent); background: rgba(138,180,248,.08); }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: var(--subtext) !important;
    border-radius: 12px 12px 0 0 !important;
    font-family: 'Google Sans', sans-serif !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#4285f4,#c58af9);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:1.1rem;">📈</div>
        <div>
            <div style="font-weight:700;font-size:1.05rem;">StockMind AI</div>
            <div style="color:#9aa0a6;font-size:.75rem;">Powered by Gemini</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔑 API Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")

    st.markdown('<div class="section-header">⚙️ Model Settings</div>', unsafe_allow_html=True)
    degree = st.slider("Polynomial Degree", min_value=1, max_value=6, value=3,
                       help="Higher = more flexible but risks overfitting")
    future_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
    gemini_model = st.selectbox("Gemini Model", [
        "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"
    ])

    st.markdown('<div class="section-header">📁 Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV / TXT",
        type=["csv", "txt"],
        help="Must contain a 'Close/Last' column"
    )

    st.markdown("---")
    st.markdown("""
    <div style="color:#9aa0a6;font-size:.78rem;line-height:1.7;">
        <b style="color:#e8eaed;">Supported columns</b><br>
        • Close/Last &nbsp;• Close &nbsp;• Price<br><br>
        Dollar signs and commas are cleaned automatically.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Stock Intelligence Studio</h1>
    <p>Polynomial regression forecasting · Gemini AI analysis · Interactive charting</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def load_data(file_obj):
    """Load and clean stock data from uploaded file."""
    try:
        content = file_obj.read()
        # Try comma, then tab, then auto
        for sep in [',', '\t', None]:
            try:
                file_obj.seek(0)
                df = pd.read_csv(io.BytesIO(content), sep=sep, engine='python') if sep else \
                     pd.read_csv(io.BytesIO(content))
                break
            except Exception:
                continue

        # Find close column
        close_col = None
        for c in df.columns:
            if c.strip().lower() in ['close/last', 'close', 'price', 'adj close', 'closing price']:
                close_col = c
                break
        if close_col is None:
            st.error(f"❌ No price column found. Columns: {list(df.columns)}")
            return None

        df = df[[close_col]].copy()
        df.columns = ['Close']
        df['Close'] = df['Close'].astype(str).str.replace(r'[$,\s]', '', regex=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(inplace=True)
        df = df.iloc[::-1].reset_index(drop=True)
        df['Day'] = np.arange(len(df))
        return df
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None


def fit_model(df, degree):
    X = df[['Day']].values
    y = df['Close'].values
    poly = PolynomialFeatures(degree=degree)
    Xp = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(Xp, y)
    y_pred = model.predict(Xp)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, poly, y_pred, mse, r2


def forecast(model, poly, last_day, future_days):
    future_X = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
    future_Xp = poly.transform(future_X)
    preds = model.predict(future_Xp)
    return future_X.flatten(), preds


def build_chart(df, y_pred, future_X, future_preds):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("", ""),
    )

    # Actual price area
    fig.add_trace(go.Scatter(
        x=df['Day'], y=df['Close'],
        name='Actual Price',
        line=dict(color='#8ab4f8', width=2),
        fill='tozeroy',
        fillcolor='rgba(138,180,248,0.07)',
    ), row=1, col=1)

    # Model fit
    fig.add_trace(go.Scatter(
        x=df['Day'], y=y_pred,
        name='Polynomial Fit',
        line=dict(color='#c58af9', width=2, dash='dot'),
    ), row=1, col=1)

    # Forecast zone
    fig.add_vrect(
        x0=df['Day'].iloc[-1], x1=future_X[-1],
        fillcolor='rgba(129,201,149,0.04)',
        line_width=0,
        row=1, col=1,
    )

    # Confidence band (±5%)
    upper = future_preds * 1.05
    lower = future_preds * 0.95
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_X, future_X[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(129,201,149,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        showlegend=True,
    ), row=1, col=1)

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_X, y=future_preds,
        name='Forecast',
        line=dict(color='#81c995', width=2.5),
        mode='lines',
    ), row=1, col=1)

    # Volume-like residuals in bottom panel
    residuals = df['Close'].values - y_pred
    colors = ['#f28b82' if r < 0 else '#81c995' for r in residuals]
    fig.add_trace(go.Bar(
        x=df['Day'], y=residuals,
        name='Residuals',
        marker_color=colors,
        opacity=0.7,
    ), row=2, col=1)

    fig.add_hline(y=0, line_color='#2e2e3a', row=2, col=1)

    fig.update_layout(
        height=560,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Google Sans', color='#9aa0a6', size=12),
        legend=dict(
            bgcolor='rgba(26,26,31,0.8)',
            bordercolor='#2e2e3a',
            borderwidth=1,
            font=dict(color='#e8eaed'),
        ),
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis2=dict(
            title='Trading Day',
            gridcolor='#1e1e26',
            zerolinecolor='#2e2e3a',
        ),
        yaxis=dict(
            title='Price (USD)',
            gridcolor='#1e1e26',
            zerolinecolor='#2e2e3a',
        ),
        yaxis2=dict(
            title='Residual',
            gridcolor='#1e1e26',
            zerolinecolor='#2e2e3a',
        ),
        hovermode='x unified',
    )
    return fig


def gemini_analysis(api_key, model_name, mse, r2, trend, last_price,
                    future_price, recent_df, degree, future_days):
    genai.configure(api_key=api_key)
    recent_str = recent_df.tail(30)[['Day', 'Close']].to_string(index=False)

    system = f"""You are StockMind, an elite quantitative analyst AI built into a professional trading platform.
Your analysis is precise, structured, and delivered with authority.
Format your response using these exact sections with emoji headers:

📊 **Trend Summary**
Brief 2-3 sentence overview of what the data shows.

📐 **Model Reliability**
Explain MSE={mse:.2f} and R²={r2:.4f} in plain language. Is this trustworthy?

⚠️ **Risk Factors**
Bullet list of overfitting risks, limitations, and caveats for degree={degree} polynomial over {future_days} days.

🔮 **{future_days}-Day Outlook**
Concrete price expectation. What range is realistic? Use the trend ({trend}) and delta.

💡 **Actionable Insight**
1-2 sentences of what a retail investor should take away from this.

Keep each section tight. Use numbers. Be direct."""

    prompt = f"""Analyze this stock model:

• MSE: {mse:.2f}
• R²: {r2:.4f}  
• Polynomial degree: {degree}
• Trend: {trend}
• Last closing price: ${last_price:.2f}
• Predicted price in {future_days} days: ${future_price:.2f}
• Change: {((future_price - last_price) / last_price * 100):+.2f}%

Recent 30-day data:
{recent_str}

Deliver a complete analysis following the format specified."""

    g_model = genai.GenerativeModel(model_name=model_name, system_instruction=system)
    chat = g_model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text


# ─────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────
if uploaded_file is None:
    # Empty state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#1a1a1f;border:1px solid #2e2e3a;border-radius:16px;padding:1.5rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:.6rem;">📁</div>
            <div style="font-weight:600;margin-bottom:.4rem;">Upload Data</div>
            <div style="color:#9aa0a6;font-size:.85rem;">Upload your CSV/TXT stock file from the sidebar</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#1a1a1f;border:1px solid #2e2e3a;border-radius:16px;padding:1.5rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:.6rem;">🧮</div>
            <div style="font-weight:600;margin-bottom:.4rem;">Model & Forecast</div>
            <div style="color:#9aa0a6;font-size:.85rem;">Polynomial regression fits your price history and projects forward</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#1a1a1f;border:1px solid #2e2e3a;border-radius:16px;padding:1.5rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:.6rem;">✨</div>
            <div style="font-weight:600;margin-bottom:.4rem;">Gemini Analysis</div>
            <div style="color:#9aa0a6;font-size:.85rem;">AI reads the model metrics and delivers structured insight</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Upload a stock CSV/TXT file from the sidebar to get started.")
    st.stop()

# ── Load ──
with st.spinner("Reading data..."):
    df = load_data(uploaded_file)

if df is None:
    st.stop()

# ── Fit ──
with st.spinner("Fitting polynomial model..."):
    model_reg, poly, y_pred, mse, r2 = fit_model(df, degree)
    future_X, future_preds = forecast(model_reg, poly, df['Day'].iloc[-1], future_days)

last_price   = df['Close'].iloc[-1]
future_price = future_preds[-1]
delta_pct    = (future_price - last_price) / last_price * 100
trend        = "upward 📈" if future_price > last_price else "downward 📉"

# ─────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Close",    f"${last_price:.2f}")
c2.metric("Forecast",      f"${future_price:.2f}", f"{delta_pct:+.2f}%")
c3.metric("R² Score",      f"{r2:.4f}",            "Model fit quality")
c4.metric("MSE",           f"{mse:,.0f}")
c5.metric("Data Points",   f"{len(df):,}")

st.markdown("")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Chart", "🤖  Gemini Analysis", "🗃️  Data"])

# ── Tab 1: Chart ──
with tab1:
    trend_pill = f'<span class="pill {"up" if delta_pct >= 0 else "down"}">{trend}</span>'
    degree_pill = f'<span class="pill blue">Degree {degree}</span>'
    days_pill = f'<span class="pill">{future_days}d forecast</span>'
    st.markdown(f"{trend_pill} {degree_pill} {days_pill}", unsafe_allow_html=True)

    fig = build_chart(df, y_pred, future_X, future_preds)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div style="color:#9aa0a6;font-size:.8rem;text-align:center;margin-top:-.5rem;">
        Shaded green zone = 30-day forecast · Dashed purple = polynomial fit · Blue band = ±5% confidence
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Gemini ──
with tab2:
    if not api_key:
        st.warning("🔑 Enter your Gemini API key in the sidebar to unlock AI analysis.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("""
            <div style="color:#9aa0a6;font-size:.9rem;">
                Gemini will analyze the model metrics, residuals, and price trajectory
                to deliver a structured financial insight report.
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            run_ai = st.button("✨ Run Analysis", use_container_width=True)

        if run_ai or st.session_state.get('ai_result'):
            if run_ai:
                with st.spinner("Gemini is thinking..."):
                    try:
                        result = gemini_analysis(
                            api_key, gemini_model, mse, r2, trend,
                            last_price, future_price, df, degree, future_days
                        )
                        st.session_state['ai_result'] = result
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")
                        st.stop()

            if 'ai_result' in st.session_state:
                st.markdown(f"""
                <div class="ai-bubble">
                    <div class="gemini-icon">
                        <svg width="18" height="18" viewBox="0 0 28 28" fill="none">
                            <path d="M14 2C7.4 2 2 7.4 2 14s5.4 12 12 12 12-5.4 12-12S20.6 2 14 2z"
                                  fill="url(#g1)"/>
                            <defs>
                                <linearGradient id="g1" x1="2" y1="2" x2="26" y2="26">
                                    <stop offset="0%" stop-color="#4285f4"/>
                                    <stop offset="100%" stop-color="#c58af9"/>
                                </linearGradient>
                            </defs>
                        </svg>
                        StockMind · Gemini
                    </div>
                    {st.session_state['ai_result'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

                # Follow-up chat
                st.markdown("---")
                st.markdown('<div class="section-header">💬 Ask a follow-up</div>', unsafe_allow_html=True)
                followup = st.text_input("", placeholder="e.g. What's the biggest risk in this forecast?")
                if st.button("Send", key="followup_btn") and followup:
                    with st.spinner("Thinking..."):
                        try:
                            genai.configure(api_key=api_key)
                            g2 = genai.GenerativeModel(model_name=gemini_model)
                            ctx = f"Context: R²={r2:.4f}, MSE={mse:.2f}, trend={trend}, last=${last_price:.2f}, forecast=${future_price:.2f}, degree={degree}\n\nPrevious analysis:\n{st.session_state['ai_result']}"
                            r2_resp = g2.generate_content(f"{ctx}\n\nUser question: {followup}")
                            st.markdown(f"""
                            <div class="ai-bubble">{r2_resp.text.replace(chr(10), '<br>')}</div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

# ── Tab 3: Data ──
with tab3:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<div class="section-header">📋 Price History</div>', unsafe_allow_html=True)
        display_df = df[['Day', 'Close']].copy()
        display_df['Model Fit'] = y_pred.round(2)
        display_df['Residual'] = (df['Close'] - y_pred).round(2)
        st.dataframe(
            display_df.style.format({'Close': '${:.2f}', 'Model Fit': '${:.2f}', 'Residual': '{:+.2f}'}),
            use_container_width=True,
            height=420,
        )

    with col_right:
        st.markdown('<div class="section-header">🔮 Forecast Table</div>', unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            'Day':  future_X,
            'Price': [f"${p:.2f}" for p in future_preds],
        })
        st.dataframe(forecast_df, use_container_width=True, height=420)

    # Download
    csv_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Analysis CSV",
        data=csv_bytes,
        file_name="stockmind_analysis.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#9aa0a6;font-size:.78rem;padding:.5rem 0 1rem;">
    StockMind AI &nbsp;·&nbsp; Polynomial regression + Gemini AI &nbsp;·&nbsp;
    <span style="color:#f28b82;">Not financial advice.</span>
    &nbsp;For educational and research use only.
</div>
""", unsafe_allow_html=True)
