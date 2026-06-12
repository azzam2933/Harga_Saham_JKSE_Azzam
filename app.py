import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Saham JKSE",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📊 Analisis Saham IDX (JKSE)")
st.markdown("---")

# Sidebar untuk input
st.sidebar.header("⚙️ Pengaturan")

# Daftar saham IDX (contoh - Anda bisa tambahkan lebih banyak)
stocks = {
    "BBCA": "PT Bank Central Asia Tbk",
    "BBRI": "PT Bank Rakyat Indonesia Tbk",
    "BMRI": "PT Bank Mandiri Tbk",
    "TLKM": "PT Telkom Indonesia Tbk",
    "ASII": "PT Astra International Tbk",
    "UNVR": "PT Unilever Indonesia Tbk",
    "ICBP": "PT Indofood CBP Sukses Makmur Tbk",
    "ADRO": "PT Adaro Energy Tbk",
    "GOTO": "PT GoTo Gojek Tokopedia Tbk",
    "BYAN": "PT Bayan Resources Tbk"
}

# Pilih saham
selected_stock = st.sidebar.selectbox(
    "Pilih Kode Saham",
    options=list(stocks.keys()),
    format_func=lambda x: f"{x} - {stocks[x]}"
)

# Pilih periode
period_options = {
    "1 Bulan": "1mo",
    "3 Bulan": "3mo",
    "6 Bulan": "6mo",
    "1 Tahun": "1y",
    "2 Tahun": "2y",
    "5 Tahun": "5y"
}
selected_period = st.sidebar.selectbox("Pilih Periode", options=list(period_options.keys()))
period = period_options[selected_period]

# Tombol refresh
if st.sidebar.button("🔄 Muat Data", type="primary"):
    st.cache_data.clear()

# Fungsi untuk mengambil data
@st.cache_data(ttl=3600)  # Cache selama 1 jam
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(f"{ticker}.JK")  # .JK untuk IDX
        df = stock.history(period=period)
        if df.empty:
            return None, "Data tidak ditemukan"
        return df, None
    except Exception as e:
        return None, str(e)

# Ambil data
with st.spinner(f"Mengambil data {selected_stock}..."):
    df_raw, error = get_stock_data(selected_stock, period)

if error:
    st.error(f"❌ Error: {error}")
    st.info("Pastikan kode saham valid atau coba lagi nanti.")
    st.stop()

if df_raw is None or df_raw.empty:
    st.error("❌ Tidak ada data untuk saham ini")
    st.stop()

# Pastikan index adalah datetime
if not isinstance(df_raw.index, pd.DatetimeIndex):
    df_raw.index = pd.to_datetime(df_raw.index)

# Tampilkan data terbaru di sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Harga Terkini")
latest = df_raw.iloc[-1]
st.sidebar.metric(
    label=f"{selected_stock}",
    value=f"Rp {latest['Close']:,.0f}",
    delta=f"{((latest['Close'] - df_raw.iloc[-2]['Close']) / df_raw.iloc[-2]['Close'] * 100):.2f}%" if len(df_raw) > 1 else None
)

# Informasi tambahan
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Info")
st.sidebar.write(f"📅 Periode: {selected_period}")
st.sidebar.write(f"📊 Total data: {len(df_raw)} hari")
st.sidebar.write(f"🕐 Update terakhir: {df_raw.index[-1].strftime('%d %B %Y')}")

# Main content
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tertinggi", f"Rp {df_raw['High'].max():,.0f}")
with col2:
    st.metric("Terendah", f"Rp {df_raw['Low'].min():,.0f}")
with col3:
    st.metric("Rata-rata", f"Rp {df_raw['Close'].mean():,.0f}")
with col4:
    st.metric("Volatilitas", f"{df_raw['Close'].pct_change().std() * 100:.2f}%")

st.markdown("---")

# Fungsi untuk membuat candlestick chart
def create_candlestick_chart(df, title):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlestick",
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        yaxis_title="Harga (Rp)",
        xaxis_title="Tanggal",
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name="Volume",
        yaxis="y2",
        marker_color='rgba(100, 100, 255, 0.5)'
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        )
    )
    
    return fig

# Tampilkan candlestick untuk periode yang dipilih
st.subheader(f"📈 Grafik Candlestick - {selected_stock}")
fig1 = create_candlestick_chart(df_raw, f"{selected_stock} - {stocks[selected_stock]} ({selected_period})")
st.plotly_chart(fig1, use_container_width=True)

# Grafik untuk 1 tahun terakhir (FIXED - menggunakan pendekatan yang benar)
st.markdown("---")
st.subheader(f"📊 Grafik Candlestick - 1 Tahun Terakhir")

# FIX: Menggunakan .loc dengan filter tanggal, bukan .last()
# Ambil 365 hari terakhir dari data
if len(df_raw) > 0:
    # Hitung tanggal cutoff (365 hari yang lalu dari tanggal terakhir)
    last_date = df_raw.index.max()
    cutoff_date = last_date - pd.Timedelta(days=365)
    
    # Filter data untuk 1 tahun terakhir
    last_year_df = df_raw[df_raw.index >= cutoff_date].copy()
    
    if len(last_year_df) > 0:
        fig2 = create_candlestick_chart(
            last_year_df, 
            f"{selected_stock} - 1 Tahun Terakhir ({last_year_df.index[0].strftime('%d %b %Y')} - {last_year_df.index[-1].strftime('%d %b %Y')})"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Tidak cukup data untuk 1 tahun terakhir")
else:
    st.warning("Data tidak tersedia")

# Moving Average options
st.markdown("---")
st.subheader("📉 Indikator Teknikal")

col_ma1, col_ma2 = st.columns(2)

with col_ma1:
    show_ma20 = st.checkbox("Tampilkan MA20 (20 hari)")
with col_ma2:
    show_ma50 = st.checkbox("Tampilkan MA50 (50 hari)")

if show_ma20 or show_ma50:
    df_ma = df_raw.copy()
    if show_ma20:
        df_ma['MA20'] = df_ma['Close'].rolling(window=20).mean()
    if show_ma50:
        df_ma['MA50'] = df_ma['Close'].rolling(window=50).mean()
    
    fig3 = go.Figure()
    
    # Candlestick
    fig3.add_trace(go.Candlestick(
        x=df_ma.index,
        open=df_ma['Open'],
        high=df_ma['High'],
        low=df_ma['Low'],
        close=df_ma['Close'],
        name="Harga",
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ))
    
    # MA20
    if show_ma20:
        fig3.add_trace(go.Scatter(
            x=df_ma.index,
            y=df_ma['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='yellow', width=1.5)
        ))
    
    # MA50
    if show_ma50:
        fig3.add_trace(go.Scatter(
            x=df_ma.index,
            y=df_ma['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='orange', width=1.5)
        ))
    
    fig3.update_layout(
        title=f"{selected_stock} dengan Moving Average",
        yaxis_title="Harga (Rp)",
        xaxis_title="Tanggal",
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# Statistik deskriptif
st.markdown("---")
st.subheader("📋 Statistik Deskriptif")

col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.write("**Statistik Harga**")
    stats_price = df_raw[['Open', 'High', 'Low', 'Close']].describe()
    st.dataframe(stats_price.style.format("{:,.0f}"), use_container_width=True)

with col_stat2:
    st.write("**Statistik Volume**")
    stats_volume = df_raw[['Volume']].describe()
    st.dataframe(stats_volume.style.format("{:,.0f}"), use_container_width=True)

# Returns analysis
st.markdown("---")
st.subheader("📈 Analisis Return")

df_raw['Daily_Return'] = df_raw['Close'].pct_change() * 100

col_ret1, col_ret2, col_ret3 = st.columns(3)

with col_ret1:
    st.metric("Rata-rata Return Harian", f"{df_raw['Daily_Return'].mean():.3f}%")
with col_ret2:
    st.metric("Return Tertinggi", f"{df_raw['Daily_Return'].max():.3f}%")
with col_ret3:
    st.metric("Return Terendah", f"{df_raw['Daily_Return'].min():.3f}%")

# Histogram return
fig_ret = go.Figure()
fig_ret.add_trace(go.Histogram(
    x=df_raw['Daily_Return'].dropna(),
    nbinsx=50,
    marker_color='skyblue',
    opacity=0.7
))

fig_ret.update_layout(
    title="Distribusi Return Harian",
    xaxis_title="Return (%)",
    yaxis_title="Frekuensi",
    template='plotly_dark',
    height=400
)

st.plotly_chart(fig_ret, use_container_width=True)

# Download data
st.markdown("---")
st.subheader("💾 Download Data")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

csv = convert_df_to_csv(df_raw)
st.download_button(
    label="📥 Download Data sebagai CSV",
    data=csv,
    file_name=f"{selected_stock}_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Data disediakan oleh Yahoo Finance | Dibuat dengan Streamlit</p>
        <p>⚠️ Disclaimer: Aplikasi ini hanya untuk tujuan edukasi. Bukan rekomendasi investasi.</p>
    </div>
    """,
    unsafe_allow_html=True
)
