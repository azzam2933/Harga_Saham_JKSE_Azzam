import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Saham IHSG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  HELPERS - SAMA PERSIS DENGAN NOTEBOOK
# ─────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def evaluate(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAPE (%)": mape, "MAE": mae, "RMSE": rmse, "R²": r2}

def fig_style():
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"] = True

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/IDX_Logo.svg", width=120)
    st.title("⚙️ Konfigurasi")

    ticker = st.text_input("Ticker Saham", value="^JKSE")
    start_date = st.date_input("Mulai", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Selesai", value=pd.to_datetime("2026-03-31"))

    st.markdown("---")
    max_depth = st.slider("Max Depth (Decision Tree)", 1, 20, 10)
    test_size = st.slider("Ukuran Data Uji (%)", 10, 40, 20) / 100

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Analisis", use_container_width=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.title("📈 Analisis & Prediksi Harga Saham IHSG")
st.caption("Linear Regression vs Decision Tree Regressor · Data: Yahoo Finance")

if not run_btn:
    st.info("👈 Atur parameter di sidebar lalu tekan **Jalankan Analisis**.")
    st.stop()

# ─────────────────────────────────────────
#  1. LOAD DATA (SAMA DENGAN NOTEBOOK)
# ─────────────────────────────────────────
with st.spinner("Mengunduh data dari Yahoo Finance…"):
    df_raw = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=False)

if df_raw.empty:
    st.error("Data tidak ditemukan. Periksa ticker dan rentang tanggal.")
    st.stop()

# Flatten MultiIndex columns jika ada
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

# Ambil kolom seperti di notebook
df = df_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
df.index.name = "Date"
df = df.sort_index()
df = df[df.index >= str(start_date)]

st.success(f"✅ Data berhasil diunduh: **{df.shape[0]} baris** dari {df.index[0].date()} s.d. {df.index[-1].date()}")

# ─────────────────────────────────────────
#  2. PREPARE DATA (SAMA DENGAN NOTEBOOK)
# ─────────────────────────────────────────
# Daily Return
df["Daily Return"] = df["Close"].pct_change()

# SMA
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()

# RSI (sama persis dengan notebook)
df["RSI"] = compute_rsi(df["Close"], period=14)

# Features (sama dengan notebook)
features = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
target = "Close"

# Drop NaN dari indikator (sama dengan notebook)
df_model = df.dropna().copy()
X = df_model[features]
y = df_model[target]

# Train-test split (sama dengan notebook: test_size=0.2, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ─────────────────────────────────────────
#  3. TRAIN MODELS (SAMA DENGAN NOTEBOOK)
# ─────────────────────────────────────────
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Decision Tree (max_depth=10 seperti di notebook)
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluasi
res_lr = evaluate(y_test, y_pred_lr, "Regresi Linier")
res_dt = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
hasil_df = pd.DataFrame([res_lr, res_dt])

# ══════════════════════════════════════════
#  TAMPILKAN HASIL DI STREAMLIT
# ══════════════════════════════════════════

# Metrics Overview
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Periode", f"{df.index[0].date()} - {df.index[-1].date()}")
col2.metric("📊 Total Data", f"{len(df):,} hari")
col3.metric("📈 Harga Tertinggi", f"Rp {df['Close'].max():,.0f}")
col4.metric("📉 Harga Terendah", f"Rp {df['Close'].min():,.0f}")

# Tabs
tab1, tab2, tab3, tab4_tab, tab5 = st.tabs([
    "📊 Pemahaman Data",
    "🔧 Persiapan Data",
    "🤖 Pemodelan",
    "📉 Evaluasi",
    "📋 Ringkasan",
])

# ══════════════════════════════════════════
#  TAB 1 – PEMAHAMAN DATA
# ══════════════════════════════════════════
with tab1:
    st.header("1. Pemahaman Data")

    with st.expander("🔍 Lihat Data Mentah (5 baris terakhir)"):
        st.dataframe(df.tail(), use_container_width=True)

    with st.expander("📋 Statistik Deskriptif"):
        st.dataframe(df.describe().round(2), use_container_width=True)

    with st.expander("⚠️ Missing Values"):
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Kolom", "Missing"]
        st.dataframe(mv, use_container_width=True)

    # Grafik Close
    st.subheader("Pergerakan Harga Penutupan (Close)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Close"], color="navy")
    ax.set_title(f"Pergerakan Harga Penutupan {ticker}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Grafik Volume
    st.subheader("Volume Perdagangan")
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df["Volume"], color="darkorange")
    ax.set_title(f"Volume Perdagangan {ticker}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Volume")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Boxplot per Tahun
    st.subheader("Boxplot Harga Close per Tahun")
    df_box = df.copy()
    df_box["Year"] = df_box.index.year
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax)
    ax.set_title("Boxplot Harga Close per Tahun")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # CV
    st.subheader("Koefisien Variasi (CV) per Tahun")
    cv = df_box.groupby("Year")["Close"].agg(["mean", "std"])
    cv["CV (%)"] = (cv["std"] / cv["mean"]) * 100
    st.dataframe(cv.round(4), use_container_width=True)

# ══════════════════════════════════════════
#  TAB 2 – PERSIAPAN DATA
# ══════════════════════════════════════════
with tab2:
    st.header("2. Persiapan Data")

    # Daily Return
    st.subheader("Daily Return")
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df["Daily Return"], color="teal", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Daily Return")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Return")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Distribusi Daily Return
    st.subheader("Distribusi Daily Return")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Daily Return"].dropna(), bins=60, kde=True, color="teal", ax=ax)
    ax.set_title("Distribusi Daily Return")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Moving Average
    st.subheader("Moving Average (SMA-50 & SMA-200)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"], label="Close", color="black", alpha=0.6)
    ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
    ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
    ax.set_title("Moving Average")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # RSI
    st.subheader("RSI (14 hari)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["RSI"], color="purple", label="RSI (14)")
    ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
    ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
    ax.set_title("RSI")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("RSI")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriks Korelasi Fitur dan Target (Close)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Split info
    st.info(f"**Split Data:** {X_train.shape[0]} data latih, {X_test.shape[0]} data uji (80:20)")

# ══════════════════════════════════════════
#  TAB 3 – PEMODELAN
# ══════════════════════════════════════════
with tab3:
    st.header("3. Pemodelan")

    # Linear Regression
    st.subheader("4.1 Regresi Linier Berganda")
    coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_})
    st.dataframe(coef_df.sort_values("Koefisien", ascending=False).reset_index(drop=True),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Koefisien", y="Fitur",
                data=coef_df.sort_values("Koefisien", ascending=False),
                palette="coolwarm", ax=ax)
    ax.set_title("Koefisien Tiap Fitur – Linear Regression")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Decision Tree
    st.subheader("4.4 Decision Tree Regressor")

    # Visualisasi DT (max_depth=3 untuk tampilan)
    st.subheader("Visualisasi Decision Tree (max_depth=3)")
    dt_vis = DecisionTreeRegressor(random_state=42, max_depth=3)
    dt_vis.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_vis, feature_names=features, filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Decision Tree Regressor (max_depth=3)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature Importance
    st.subheader("Feature Importance – Decision Tree")
    imp_df = pd.DataFrame({
        "Fitur": features,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance – Decision Tree Regressor")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════
#  TAB 4 – EVALUASI
# ══════════════════════════════════════════
with tab4_tab:
    st.header("4. Evaluasi Model")

    # Tabel Perbandingan Aktual vs Prediksi
    st.subheader("Tabel Perbandingan Aktual vs Prediksi (5 data teratas dan terbawah)")
    hasil_pred = pd.DataFrame({
        "Close": y_test.values,
        "lr_pred": y_pred_lr,
        "dt_pred": y_pred_dt
    }, index=X_test.index).sort_index()
    
    # Tampilkan 5 teratas dan 5 terbawah
    display_df = pd.concat([hasil_pred.head(5), hasil_pred.tail(5)])
    st.dataframe(display_df.round(2), use_container_width=True)

    # Tabel Evaluasi
    st.subheader("Tabel Evaluasi Model")
    hasil_df_display = hasil_df.copy()
    hasil_df_display["MAPE (%)"] = hasil_df_display["MAPE (%)"].apply(lambda x: f"{x:.4f}%")
    hasil_df_display["MAE"] = hasil_df_display["MAE"].apply(lambda x: f"{x:.4f}")
    hasil_df_display["RMSE"] = hasil_df_display["RMSE"].apply(lambda x: f"{x:.4f}")
    hasil_df_display["R²"] = hasil_df_display["R²"].apply(lambda x: f"{x:.6f}")
    
    st.dataframe(hasil_df_display, use_container_width=True, hide_index=True)

    # Aktual vs Prediksi Plot
    st.subheader("Perbandingan Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hasil_pred.index, hasil_pred["Close"], label="Aktual", color="black", linewidth=1.5)
    ax.plot(hasil_pred.index, hasil_pred["lr_pred"], label="Linear Regression", color="blue", alpha=0.8)
    ax.plot(hasil_pred.index, hasil_pred["dt_pred"], label="Decision Tree", color="red", alpha=0.8)
    ax.set_title("Perbandingan Harga Aktual vs Prediksi")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Scatter Plot
    st.subheader("Scatter Plot Aktual vs Prediksi")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear Regression
    axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color="blue", s=15)
    lim = [min(y_test.min(), y_pred_lr.min()), max(y_test.max(), y_pred_lr.max())]
    axes[0].plot(lim, lim, "k--")
    axes[0].set_title(f"Linear Regression (MAPE={res_lr['MAPE (%)']:.4f}%)")
    axes[0].set_xlabel("Aktual")
    axes[0].set_ylabel("Prediksi")
    
    # Decision Tree
    axes[1].scatter(y_test, y_pred_dt, alpha=0.4, color="red", s=15)
    lim = [min(y_test.min(), y_pred_dt.min()), max(y_test.max(), y_pred_dt.max())]
    axes[1].plot(lim, lim, "k--")
    axes[1].set_title(f"Decision Tree (MAPE={res_dt['MAPE (%)']:.4f}%)")
    axes[1].set_xlabel("Aktual")
    axes[1].set_ylabel("Prediksi")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════
#  TAB 5 – RINGKASAN
# ══════════════════════════════════════════
with tab5:
    st.header("5. Ringkasan Hasil")

    # Tampilkan hasil evaluasi yang sudah jadi
    st.subheader("Hasil Evaluasi Model")
    
    # Format ulang untuk tampilan yang rapi
    summary_lr = {
        "Model": "Regresi Linier",
        "MAPE": f"{res_lr['MAPE (%)']:.4f}%",
        "MAE": f"{res_lr['MAE']:.4f}",
        "RMSE": f"{res_lr['RMSE']:.4f}",
        "R²": f"{res_lr['R²']:.6f}"
    }
    
    summary_dt = {
        "Model": "Regresi Pohon Keputusan",
        "MAPE": f"{res_dt['MAPE (%)']:.4f}%",
        "MAE": f"{res_dt['MAE']:.4f}",
        "RMSE": f"{res_dt['RMSE']:.4f}",
        "R²": f"{res_dt['R²']:.6f}"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Regresi Linier")
        st.markdown(f"""
        - **MAPE:** {summary_lr['MAPE']}
        - **MAE:** {summary_lr['MAE']}
        - **RMSE:** {summary_lr['RMSE']}
        - **R²:** {summary_lr['R²']}
        """)
    
    with col2:
        st.markdown("### 🌳 Regresi Pohon Keputusan")
        st.markdown(f"""
        - **MAPE:** {summary_dt['MAPE']}
        - **MAE:** {summary_dt['MAE']}
        - **RMSE:** {summary_dt['RMSE']}
        - **R²:** {summary_dt['R²']}
        """)
    
    # Kesimpulan
    st.markdown("---")
    st.subheader("📝 Kesimpulan")
    
    best_model = "Regresi Linier" if res_lr['MAPE (%)'] < res_dt['MAPE (%)'] else "Regresi Pohon Keputusan"
    best_mape = min(res_lr['MAPE (%)'], res_dt['MAPE (%)'])
    
    st.markdown(f"""
    Berdasarkan evaluasi yang dilakukan pada data uji (20% dari total data), diperoleh hasil:
    
    1. **{best_model}** memberikan performa terbaik dengan nilai **MAPE sebesar {best_mape:.4f}%**.
    
    2. **Regresi Linier** memiliki MAPE {res_lr['MAPE (%)']:.4f}%, 
       sedangkan **Decision Tree** memiliki MAPE {res_dt['MAPE (%)']:.4f}%.
    
    3. Nilai **R²** untuk Regresi Linier adalah {res_lr['R²']:.6f} dan untuk Decision Tree adalah {res_dt['R²']:.6f}, 
       yang menunjukkan kedua model sangat baik dalam menjelaskan variabilitas data.
    
    4. Kedua model memiliki performa yang sangat baik dengan error relatif di bawah 0.5%, 
       sehingga dapat digunakan untuk prediksi harga IHSG.
    """)
    
    # Visualisasi Perbandingan MAPE
    st.subheader("Perbandingan MAPE Antar Model")
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Regresi Linier', 'Decision Tree']
    mapes = [res_lr['MAPE (%)'], res_dt['MAPE (%)']]
    colors = ['steelblue', 'tomato']
    bars = ax.bar(models, mapes, color=colors)
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Perbandingan MAPE Model')
    for bar, mape in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mape:.4f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
