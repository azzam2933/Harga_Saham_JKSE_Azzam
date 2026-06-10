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
#  HELPERS
# ─────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def evaluate(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAPE (%)": round(mape, 4),
            "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)}


def fig_style():
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"] = True


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("Foto Berwarna.jpg", width=110)
    st.title("Muhamad Azzam Khoiri")
    st.text("<h2>Universitas Gunadarma<h2>")

    ticker = st.text_input("Ticker Saham", value="^JKSE")
    start_date = st.date_input("Mulai", value=pd.to_datetime("2020-01-01"))
    end_date   = st.date_input("Selesai", value=pd.to_datetime("2026-03-31"))

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
#  1. LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Mengunduh data dari Yahoo Finance…"):
    df_raw = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=False)

if df_raw.empty:
    st.error("Data tidak ditemukan. Periksa ticker dan rentang tanggal.")
    st.stop()

if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)

df = df_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
df.index.name = "Date"
df = df.sort_index()
df = df[df.index >= str(start_date)]

st.success(f"✅ Data berhasil diunduh: **{df.shape[0]} baris** dari {df.index[0].date()} s.d. {df.index[-1].date()}")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris", df.shape[0])
    col2.metric("Harga Close Terakhir", f"{df['Close'].iloc[-1]:,.0f}")
    col3.metric("Harga Close Max", f"{df['Close'].max():,.0f}")
    col4.metric("Harga Close Min", f"{df['Close'].min():,.0f}")

    with st.expander("🔍 Lihat Data Mentah"):
        st.dataframe(df, use_container_width=True)

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
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Grafik Volume
    st.subheader("Volume Perdagangan")
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df["Volume"], color="darkorange")
    ax.set_title(f"Volume Perdagangan {ticker}")
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Volume")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Candlestick (setahun terakhir)
    st.subheader("Candlestick – Setahun Terakhir")
    last_year = df[df.index >= df.index.max() - pd.Timedelta(days=365)].copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    for date, row in last_year.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.vlines(date, row["Low"], row["High"], color=color, linewidth=1)
        body_low  = min(row["Open"], row["Close"])
        body_high = max(row["Open"], row["Close"])
        ax.add_patch(plt.Rectangle(
            (date - pd.Timedelta(days=0.3), body_low),
            pd.Timedelta(days=0.6), body_high - body_low,
            facecolor=color, edgecolor=color))
    ax.set_title("Candlestick – Setahun Terakhir")
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Boxplot per Tahun
    st.subheader("Boxplot Harga Close per Tahun")
    df_box = df.copy(); df_box["Year"] = df_box.index.year
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax)
    ax.set_title("Boxplot Harga Close per Tahun")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # CV
    st.subheader("Koefisien Variasi (CV) per Tahun")
    cv = df_box.groupby("Year")["Close"].agg(["mean", "std"])
    cv["CV (%)"] = (cv["std"] / cv["mean"]) * 100
    st.dataframe(cv.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cv.index.astype(str), cv["CV (%)"], color="steelblue", edgecolor="black")
    for i, v in enumerate(cv["CV (%)"]):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=10)
    ax.set_title("Koefisien Variasi (CV) per Tahun")
    ax.set_xlabel("Tahun"); ax.set_ylabel("CV (%)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════
#  TAB 2 – PERSIAPAN DATA
# ══════════════════════════════════════════
with tab2:
    st.header("2. Persiapan Data")

    df = df.dropna()

    # Daily Return
    df["Daily Return"] = df["Close"].pct_change()

    st.subheader("Daily Return")
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df["Daily Return"], color="teal", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Daily Return"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Return")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Distribusi Daily Return")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Daily Return"].dropna(), bins=60, kde=True, color="teal", ax=ax)
    ax.set_title("Distribusi Daily Return")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Moving Averages
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    st.subheader("Moving Average (SMA-50 & SMA-200)")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"],    label="Close",       color="black",  alpha=0.6)
    ax.plot(df.index, df["SMA_50"],   label="SMA 50 hari", color="blue")
    ax.plot(df.index, df["SMA_200"],  label="SMA 200 hari",color="red")
    ax.set_title("Moving Average"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # RSI
    df["RSI"] = compute_rsi(df["Close"], 14)

    st.subheader("RSI (14 hari)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["RSI"], color="purple", label="RSI (14)")
    ax.axhline(70, linestyle="--", color="red",   label="Overbought (70)")
    ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    ax.axhline(50, linestyle=":",  color="gray",  alpha=0.6)
    ax.set_title("RSI"); ax.set_xlabel("Tanggal"); ax.set_ylabel("RSI")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature selection
    features = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
    target   = "Close"
    df_model = df.dropna().copy()
    X = df_model[features]; y = df_model[target]

    st.subheader("Heatmap Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriks Korelasi Fitur dan Target (Close)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True)

    col1, col2 = st.columns(2)
    col1.metric("Data Latih", X_train.shape[0])
    col2.metric("Data Uji",   X_test.shape[0])

    # Simpan ke session state agar tab lain bisa pakai
    st.session_state["data"] = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": features,
    }


# ══════════════════════════════════════════
#  TAB 3 – PEMODELAN
# ══════════════════════════════════════════
with tab3:
    st.header("3. Pemodelan")

    if "data" not in st.session_state:
        st.warning("Jalankan tab Persiapan Data terlebih dahulu.")
        st.stop()

    d = st.session_state["data"]
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    features = d["features"]

    # Linear Regression
    st.subheader("4.1 Regresi Linier Berganda")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_})
    st.dataframe(coef_df.sort_values("Koefisien", ascending=False).reset_index(drop=True),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x="Koefisien", y="Fitur",
                data=coef_df.sort_values("Koefisien", ascending=False),
                palette="coolwarm", ax=ax)
    ax.set_title("Koefisien – Linear Regression")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Decision Tree
    st.subheader("4.4 Decision Tree Regressor")
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Visualisasi DT (max_depth=3 untuk tampilan)
    st.subheader("Visualisasi Decision Tree (max_depth=3)")
    dt_vis = DecisionTreeRegressor(random_state=42, max_depth=3)
    dt_vis.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt_vis, feature_names=features, filled=True, rounded=True,
              fontsize=8, ax=ax)
    ax.set_title("Decision Tree Regressor (max_depth=3)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature Importance
    st.subheader("Feature Importance – Decision Tree")
    imp_df = pd.DataFrame({
        "Fitur": features,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance – Decision Tree")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    # Simpan prediksi ke session state
    st.session_state["preds"] = {
        "y_test": y_test, "y_pred_lr": y_pred_lr, "y_pred_dt": y_pred_dt,
        "X_test": X_test,
    }


# ══════════════════════════════════════════
#  TAB 4 – EVALUASI
# ══════════════════════════════════════════
with tab4:
    st.header("4. Evaluasi Model")

    if "preds" not in st.session_state:
        st.warning("Jalankan tab Pemodelan terlebih dahulu.")
        st.stop()

    p = st.session_state["preds"]
    y_test    = p["y_test"]
    y_pred_lr = p["y_pred_lr"]
    y_pred_dt = p["y_pred_dt"]
    test_idx  = p["X_test"].index

    res_lr = evaluate(y_test, y_pred_lr, "Regresi Linier")
    res_dt = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
    hasil  = pd.DataFrame([res_lr, res_dt])

    st.subheader("Tabel Metrik Evaluasi")
    st.dataframe(hasil.set_index("Model"), use_container_width=True)

    # Aktual vs Prediksi gabungan
    st.subheader("Aktual vs Prediksi (Data Uji)")
    plot_df = pd.DataFrame({
        "Aktual": y_test,
        "Linear Regression": y_pred_lr,
        "Decision Tree": y_pred_dt,
    }, index=test_idx).sort_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df.index, plot_df["Aktual"],           label="Aktual",           color="black", linewidth=1.5)
    ax.plot(plot_df.index, plot_df["Linear Regression"],label="Linear Regression",color="blue",  alpha=0.8)
    ax.plot(plot_df.index, plot_df["Decision Tree"],    label="Decision Tree",    color="red",   alpha=0.8)
    ax.set_title("Perbandingan Harga Aktual vs Prediksi")
    ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Scatter plot
    st.subheader("Scatter Plot Aktual vs Prediksi")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_p, lbl, color in [
        (axes[0], y_pred_lr, "Linear Regression", "blue"),
        (axes[1], y_pred_dt, "Decision Tree",     "red"),
    ]:
        ax.scatter(y_test, y_p, alpha=0.4, color=color, s=15)
        lim = [y_test.min(), y_test.max()]
        ax.plot(lim, lim, "k--")
        r = res_lr if lbl == "Linear Regression" else res_dt
        ax.set_title(f"{lbl} (MAPE={r['MAPE (%)']:.2f}%)")
        ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Distribusi Residual
    st.subheader("Distribusi Residual")
    residual_lr = y_test.values - y_pred_lr
    residual_dt = y_test.values - y_pred_dt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, res, lbl, color in [
        (axes[0], residual_lr, "Linear Regression", "blue"),
        (axes[1], residual_dt, "Decision Tree",     "red"),
    ]:
        sns.histplot(res, bins=40, kde=True, color=color, ax=ax)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"Residual – {lbl}")
        ax.set_xlabel("Residual")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Perbandingan MAPE
    st.subheader("Perbandingan MAPE")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Model", y="MAPE (%)", data=hasil,
                palette=["steelblue", "tomato"], ax=ax)
    for i, v in enumerate(hasil["MAPE (%)"]):
        ax.text(i, v + 0.02, f"{v:.4f}%", ha="center", fontweight="bold")
    ax.set_title("Perbandingan MAPE antar Model")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Perbandingan semua metrik
    st.subheader("Perbandingan Semua Metrik")
    hasil_melt = hasil.melt(id_vars="Model",
                             value_vars=["MAPE (%)", "MAE", "RMSE"],
                             var_name="Metrik", value_name="Nilai")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Metrik", y="Nilai", hue="Model", data=hasil_melt,
                palette=["steelblue", "tomato"], ax=ax)
    ax.set_title("Perbandingan Metrik Error (MAPE, MAE, RMSE)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.session_state["hasil"] = hasil


# ══════════════════════════════════════════
#  TAB 5 – RINGKASAN
# ══════════════════════════════════════════
with tab5:
    st.header("5. Ringkasan Hasil")

    if "hasil" not in st.session_state:
        st.warning("Jalankan evaluasi terlebih dahulu.")
        st.stop()

    hasil = st.session_state["hasil"]

    best = hasil.loc[hasil["MAPE (%)"].idxmin(), "Model"]
    best_mape = hasil["MAPE (%)"].min()

    st.success(f"🏆 **Model Terbaik:** {best} dengan MAPE **{best_mape:.4f}%**")

    col1, col2 = st.columns(2)
    for _, row in hasil.iterrows():
        c = col1 if row["Model"] == "Regresi Linier" else col2
        c.markdown(f"### {row['Model']}")
        c.metric("MAPE (%)", f"{row['MAPE (%)']:.4f}")
        c.metric("MAE",       f"{row['MAE']:,.2f}")
        c.metric("RMSE",      f"{row['RMSE']:,.2f}")
        c.metric("R²",        f"{row['R²']:.4f}")

    st.markdown("---")
    st.markdown("""
    **Catatan Interpretasi:**
    - **MAPE** → semakin kecil semakin baik (error relatif dalam %)
    - **MAE / RMSE** → semakin kecil semakin baik (error absolut)
    - **R²** → semakin mendekati 1 semakin baik (proporsi varians yang dijelaskan model)
    """)
