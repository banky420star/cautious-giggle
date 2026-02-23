import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="NeuroTrader v3.1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for iOS 18 Glassmorphism Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #000000;
        color: #FFFFFF;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1e2f 0%, #000000 100%);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 600;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .status-badge {
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(0, 255, 128, 0.2);
        color: #00ff80;
        border: 1px solid rgba(0, 255, 128, 0.3);
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -1px;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading Functions
def load_data():
    try:
        equity_df = pd.read_csv("data/equity_curve.csv")
        pnl_df = pd.read_csv("data/pnl_per_pair.csv")
        with open("data/metrics.json", "r") as f:
            metrics = json.load(f)
        return equity_df, pnl_df, metrics
    except:
        return None, None, None

equity_df, pnl_df, metrics = load_data()

# Header
cols = st.columns([2, 1])
with cols[0]:
    st.markdown("<h1 style='margin-bottom:0;'>NeuroTrader <span style='color:#3a7bd5;'>v3.1</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.5);'>Self-Evolving RL Trading System</p>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("<div style='text-align:right; margin-top:20px;'><span class='status-badge'>● LIVE ENGINE ACTIVE</span></div>", unsafe_allow_html=True)

st.markdown("---")

if metrics:
    # Top Metrics Row
    m_cols = st.columns(4)
    with m_cols[0]:
        st.markdown(f"""<div class='glass-card'><p class='metric-label'>Total Equity</p><p class='metric-value'>${equity_df['equity'].iloc[-1]:,.2f}</p></div>""", unsafe_allow_html=True)
    with m_cols[1]:
        st.markdown(f"""<div class='glass-card'><p class='metric-label'>Daily PnL</p><p class='metric-value' style='background: linear-gradient(90deg, #00FF80, #60FFB0); -webkit-text-fill-color: transparent;'>+${metrics['total_profit']:,.2f}</p></div>""", unsafe_allow_html=True)
    with m_cols[2]:
        st.markdown(f"""<div class='glass-card'><p class='metric-label'>Sharpe Ratio</p><p class='metric-value'>{metrics['sharpe_ratio']}</p></div>""", unsafe_allow_html=True)
    with m_cols[3]:
        st.markdown(f"""<div class='glass-card'><p class='metric-label'>Max Drawdown</p><p class='metric-value' style='background: linear-gradient(90deg, #FF5050, #FF0000); -webkit-text-fill-color: transparent;'>{metrics['max_drawdown']*100:.2f}%</p></div>""", unsafe_allow_html=True)

    # Charts Row
    c_cols = st.columns([2, 1])
    
    with c_cols[0]:
        st.markdown("<div class='glass-card'><h3>Master Equity Curve</h3>", unsafe_allow_html=True)
        fig = px.line(equity_df, x='timestamp', y='equity', 
                     color_discrete_sequence=['#00d2ff'],
                     template="plotly_dark")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_cols[1]:
        st.markdown("<div class='glass-card'><h3>Symbol Distribution</h3>", unsafe_allow_html=True)
        fig_pie = px.pie(pnl_df, values='pnl', names='symbol', 
                        hole=.6,
                        color_discrete_sequence=px.colors.sequential.GnBu)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Per-Symbol Performance Table
    st.markdown("<div class='glass-card'><h3>Active Symbol Pulse</h3>", unsafe_allow_html=True)
    
    # Custom Table
    st.dataframe(
        pnl_df.style.gradient_background(subset=['pnl', 'win_rate'], cmap='GnBu'),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Waiting for data from NeuroTrader Engine... (Run full_launch.sh first)")
    if st.button("Generate Initial Mock Data"):
        os.system("python3 update_dashboard_data.py --once") 
        st.rerun()

# Sidebar (App Settings)
with st.sidebar:
    st.title("Settings")
    st.toggle("Turbo Evolution Mode", value=True)
    st.slider("Risk Tolerance", 0.1, 5.0, 1.0)
    st.button("Force Global Rollback")
