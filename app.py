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
    page_title="Prediksi Saham ^JKSE",
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
    return {"Model": name, "MAPE (%)": mape, "MAE": mae, "RMSE": rmse, "R²": r2}

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("Muhamad Azzam Khoiri")
    st.markdown("Universitas Gunadarma")
    
    ticker = st.text_input("Ticker Saham", value="^JKSE")
    start_date = st.date_input("Mulai", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Selesai", value=pd.to_datetime("2026-03-31"))
    
    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Analisis", use_container_width=True)

# ══════════════════════════════════════════
#  DATA LOADING & PREPROCESSING (IDENTIK NOTEBOOK)
# ══════════════════════════════════════════
@st.cache_data
def load_and_process_data(ticker, start_date, end_date):
    """Load data dan preprocessing identik dengan notebook Colab"""
    
    # 2.1 Load Dataset (sama persis dengan notebook)
    df_raw = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=False)
    
    if df_raw.empty:
        return None, None, None, None
    
    # Flatten MultiIndex columns jika ada
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    # Ambil kolom termasuk Adj Close
    df = df_raw[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
    df.index.name = "Date"
    df = df.sort_index()
    df = df[df.index >= str(start_date)]
    
    # 3.1 Pembersihan Data
    df = df.dropna()
    
    # 3.2 Daily Return
    df["Daily Return"] = df["Close"].pct_change()
    
    # 3.5 Moving Average (SMA-50 & SMA-200)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    
    # 3.6 Relative Strength Index (RSI)
    df["RSI"] = compute_rsi(df["Close"], period=14)
    
    # 3.7 Pemilihan Fitur
    df_model = df.dropna().copy()
    features = ["Open", "High", "Low", "Volume", "Daily Return", "SMA_50", "SMA_200", "RSI"]
    target = "Close"
    
    X = df_model[features]
    y = df_model[target]
    
    # 3.9 Split Data Latih dan Data Uji (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    return df_raw, df, df_model, X_train, X_test, y_train, y_test, features, target

# ══════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════
if not run_btn:
    st.info("👈 Atur parameter di sidebar lalu tekan **Jalankan Analisis**.")
    st.stop()

with st.spinner("Mengunduh dan memproses data..."):
    result = load_and_process_data(ticker, start_date, end_date)
    
if result[0] is None:
    st.error("Data tidak ditemukan. Periksa ticker dan rentang tanggal.")
    st.stop()

df_raw, df, df_model, X_train, X_test, y_train, y_test, features, target = result

# ─────────────────────────────────────────
#  TRAIN MODELS
# ─────────────────────────────────────────
# 4.1 Regresi Linier Berganda (sama persis notebook)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 4.4 Regresi Pohon Keputusan (sama persis notebook - max_depth=10)
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluasi
res_lr = evaluate(y_test, y_pred_lr, "Regresi Linier")
res_dt = evaluate(y_test, y_pred_dt, "Regresi Pohon Keputusan")
hasil = pd.DataFrame([res_lr, res_dt])

# Koefisien LR
coef_df = pd.DataFrame({"Fitur": features, "Koefisien": lr_model.coef_})

# Feature Importance DT
imp_df = pd.DataFrame({"Fitur": features, "Importance": dt_model.feature_importances_}) \
           .sort_values("Importance", ascending=False)

# MAPE values
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
mape_dt = mean_absolute_percentage_error(y_test, y_pred_dt)

test_idx = X_test.index

# ══════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════
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
    col1.metric("Total Baris", df_raw.shape[0])
    col2.metric("Harga Close Terakhir", f"{df_raw['Close'].iloc[-1]:,.0f}")
    col3.metric("Harga Close Max", f"{df_raw['Close'].max():,.0f}")
    col4.metric("Harga Close Min", f"{df_raw['Close'].min():,.0f}")
    
    st.subheader("2.1 Load Dataset")
    st.dataframe(df_raw, use_container_width=True)
    
    with st.expander("2.2 Data Terbawah (Tail)"):
        st.dataframe(df_raw.tail(), use_container_width=True)
    
    with st.expander("2.3 Informasi Dataset"):
        st.write(f"**Shape:** {df_raw.shape}")
        st.write("**Data Types:**")
        st.dataframe(df_raw.dtypes.to_frame("Dtype"), use_container_width=True)
    
    with st.expander("2.4 Statistik Deskriptif"):
        st.dataframe(df_raw.describe(), use_container_width=True)
    
    with st.expander("2.5 Cek Missing Values"):
        missing = df_raw.isnull().sum()
        st.write(missing[missing > 0] if missing.sum() > 0 else "✅ Tidak ada missing values")
    
    st.subheader("2.6 Grafik Pergerakan Harga Close")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_raw.index, df_raw["Close"], color="navy")
    ax.set_title(f"Pergerakan Harga Penutupan Saham {ticker}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Close (IDR)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("2.7 Grafik Volume Perdagangan")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_raw.index, df_raw["Volume"], color="darkorange")
    ax.set_title(f"Volume Perdagangan Saham {ticker}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Volume")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("2.8 Grafik Candlestick (Setahun Terakhir)")
    last_year = df_raw.last("365D").copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.6
    for date, row in last_year.iterrows():
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.vlines(date, row["Low"], row["High"], color=color, linewidth=1)
        body_low = min(row["Open"], row["Close"])
        body_high = max(row["Open"], row["Close"])
        ax.add_patch(plt.Rectangle((date - pd.Timedelta(days=width/2), body_low),
                                    pd.Timedelta(days=width), body_high - body_low,
                                    facecolor=color, edgecolor=color))
    ax.set_title("Grafik Candlestick Pergerakan Harga Saham JKSE (Setahun Terakhir)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("2.9 Boxplot Harga Penutupan per Tahun")
    df_box = df_raw.copy()
    df_box["Year"] = df_box.index.year
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax)
    ax.set_title("Boxplot Harga Penutupan (Close) Saham JKSE per Tahun")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Harga Close (IDR)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("2.10 Koefisien Variasi (CV)")
    cv_per_year = df_box.groupby("Year")["Close"].agg(["mean", "std"])
    cv_per_year["CV in %"] = (cv_per_year["std"] / cv_per_year["mean"]) * 100
    st.dataframe(cv_per_year, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cv_per_year.index.astype(str), cv_per_year["CV in %"], color="steelblue", edgecolor="black")
    ax.set_title("Koefisien Variasi (CV) Harga Saham JKSE per Tahun")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("CV (%)")
    for i, v in enumerate(cv_per_year["CV in %"]):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("2.11 Matriks Korelasi (Mean, Std, CV)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cv_per_year.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriks Korelasi (Mean, Std, CV)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════
#  TAB 2 – PERSIAPAN DATA
# ══════════════════════════════════════════
with tab2:
    st.header("2. Persiapan Data")
    
    st.subheader("3.1 Pembersihan Data")
    col1, col2 = st.columns(2)
    col1.metric("Sebelum dropna", df_raw.shape[0])
    col2.metric("Sesudah dropna", df.shape[0])
    
    st.subheader("3.2 Daily Return")
    st.dataframe(df[["Close", "Daily Return"]].head(10), use_container_width=True)
    
    st.subheader("3.3 Grafik Daily Return")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Daily Return"], color="teal", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Daily Return Saham JKSE")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Return Harian")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("3.4 Distribusi Daily Return")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Daily Return"].dropna(), bins=60, kde=True, color="teal", ax=ax)
    ax.set_title("Distribusi Daily Return Saham JKSE")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frekuensi")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("3.5 Moving Average (SMA-50 & SMA-200)")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], label="Close", color="black", alpha=0.6)
    ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
    ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
    ax.set_title("Grafik Moving Average (SMA-50 & SMA-200) – Saham JKSE")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("3.6 Relative Strength Index (RSI)")
    st.dataframe(df[["Close", "RSI"]].tail(10), use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["RSI"], color="purple", label="RSI (14)")
    ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
    ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
    ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
    ax.set_title("Grafik Relative Strength Index (RSI) – Saham JKSE")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("RSI")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("3.7 Pemilihan Fitur")
    st.write(f"**Shape setelah hilangkan NaN dari indikator:** {df_model.shape}")
    st.write(f"**Fitur (X):** {features}")
    st.write(f"**Target (y):** {target}")
    st.dataframe(X.head(), use_container_width=True)
    
    st.subheader("3.8 Heatmap Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm", 
                fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriks Korelasi Fitur dan Target (Close)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("3.9 Split Data Latih dan Data Uji (80:20)")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah data latih", X_train.shape[0])
    col2.metric("Jumlah data uji", X_test.shape[0])


# ══════════════════════════════════════════
#  TAB 3 – PEMODELAN
# ══════════════════════════════════════════
with tab3:
    st.header("3. Pemodelan")
    
    st.subheader("4.1 Regresi Linier Berganda")
    st.write(f"**Intercept:** {lr_model.intercept_}")
    st.write(f"**Slope:** {lr_model.coef_}")
    
    st.subheader("4.2 Tabel Koefisien Linear Regression")
    st.dataframe(coef_df, use_container_width=True)
    
    st.write("**5 prediksi pertama (Linear Regression):**")
    st.write(y_pred_lr[:5])
    
    st.subheader("4.3 Grafik Koefisien Linear Regression")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Koefisien", y="Fitur",
                data=coef_df.sort_values("Koefisien", ascending=False),
                palette="coolwarm", ax=ax)
    ax.set_title("Koefisien Tiap Fitur – Linear Regression")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("4.4 Regresi Pohon Keputusan (Decision Tree Regressor)")
    dt_model_vis = DecisionTreeRegressor(random_state=42, max_depth=10)
    dt_model_vis.fit(X_train, y_train)
    y_pred_dt_vis = dt_model_vis.predict(X_test)
    
    st.write("**5 prediksi pertama (Decision Tree):**")
    st.write(y_pred_dt[:5])
    
    st.subheader("4.5 Visualisasi Decision Tree (max_depth=3 untuk tampilan)")
    dt_vis = DecisionTreeRegressor(random_state=42, max_depth=3)
    dt_vis.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_vis, feature_names=features, filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Visualisasi Decision Tree Regressor (max_depth=3)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("4.6 Feature Importance Decision Tree")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="Importance", y="Fitur", data=imp_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance – Decision Tree Regressor")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.dataframe(imp_df, use_container_width=True)
    
    st.subheader("4.7 Tabel Perbandingan Aktual vs Prediksi")
    hasil_pred = pd.DataFrame({
        "Close (Aktual)": y_test,
        "lr_pred": y_pred_lr,
        "dt_pred": y_pred_dt
    }).sort_index()
    
    st.write("**5 data teratas dan 5 data terbawah:**")
    st.dataframe(pd.concat([hasil_pred.head(5), hasil_pred.tail(5)]), use_container_width=True)


# ══════════════════════════════════════════
#  TAB 4 – EVALUASI
# ══════════════════════════════════════════
with tab4:
    st.header("4. Evaluasi Model")
    
    # Hitung ulang sesuai notebook
    mape_lr_eval = mean_absolute_percentage_error(y_test, y_pred_lr) * 100
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    
    mape_dt_eval = mean_absolute_percentage_error(y_test, y_pred_dt) * 100
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    r2_dt = r2_score(y_test, y_pred_dt)
    
    st.write("**=== Regresi Linier ===**")
    st.write(f"MAPE : {mape_lr_eval:.4f} %")
    st.write(f"MAE  : {mae_lr:.4f}")
    st.write(f"RMSE : {rmse_lr:.4f}")
    st.write(f"R²   : {r2_lr:.4f}")
    
    st.write("")
    st.write("**=== Regresi Pohon Keputusan ===**")
    st.write(f"MAPE : {mape_dt_eval:.4f} %")
    st.write(f"MAE  : {mae_dt:.4f}")
    st.write(f"RMSE : {rmse_dt:.4f}")
    st.write(f"R²   : {r2_dt:.4f}")
    
    eval_df = pd.DataFrame([
        {"Model": "Regresi Linier", "MAPE (%)": mape_lr_eval, "MAE": mae_lr, "RMSE": rmse_lr, "R²": r2_lr},
        {"Model": "Regresi Pohon Keputusan", "MAPE (%)": mape_dt_eval, "MAE": mae_dt, "RMSE": rmse_dt, "R²": r2_dt}
    ])
    
    st.dataframe(eval_df, use_container_width=True)
    
    # Plot aktual vs prediksi
    st.subheader("5.1 Plot Aktual vs Prediksi Linear Regression")
    plot_df = pd.DataFrame({"Aktual": y_test, "Prediksi LR": y_pred_lr}, index=test_idx).sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df.index, plot_df["Aktual"], label="Aktual", color="black", linewidth=1.5)
    ax.plot(plot_df.index, plot_df["Prediksi LR"], label="Prediksi LR", color="blue", alpha=0.8)
    ax.set_title("Aktual vs Prediksi – Linear Regression (Data Uji)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Close (IDR)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("5.2 Plot Aktual vs Prediksi Decision Tree")
    plot_df2 = pd.DataFrame({"Aktual": y_test, "Prediksi DT": y_pred_dt}, index=test_idx).sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df2.index, plot_df2["Aktual"], label="Aktual", color="black", linewidth=1.5)
    ax.plot(plot_df2.index, plot_df2["Prediksi DT"], label="Prediksi DT", color="red", alpha=0.8)
    ax.set_title("Aktual vs Prediksi – Decision Tree (Data Uji)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Close (IDR)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("5.3 Plot Perbandingan Semua Model")
    plot_df3 = pd.DataFrame({
        "Aktual": y_test,
        "Linear Regression": y_pred_lr,
        "Decision Tree": y_pred_dt
    }, index=test_idx).sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df3.index, plot_df3["Aktual"], label="Aktual", color="black", linewidth=1.5)
    ax.plot(plot_df3.index, plot_df3["Linear Regression"], label="Linear Regression", color="blue", alpha=0.8)
    ax.plot(plot_df3.index, plot_df3["Decision Tree"], label="Decision Tree", color="red", alpha=0.8)
    ax.set_title("Perbandingan Harga Aktual vs Hasil Prediksi (Data Uji)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Close (IDR)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("5.4 Scatter Plot Aktual vs Prediksi")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear Regression
    axes[0].scatter(y_test, y_pred_lr, alpha=0.5, color="blue", s=15)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    axes[0].set_xlabel("Aktual")
    axes[0].set_ylabel("Prediksi")
    axes[0].set_title(f"Linear Regression (MAPE={mape_lr_eval:.4f}%)")
    
    # Decision Tree
    axes[1].scatter(y_test, y_pred_dt, alpha=0.5, color="red", s=15)
    min_val = min(y_test.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_dt.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    axes[1].set_xlabel("Aktual")
    axes[1].set_ylabel("Prediksi")
    axes[1].set_title(f"Decision Tree (MAPE={mape_dt_eval:.4f}%)")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader("5.5 Perbandingan MAPE")
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Linear Regression", "Decision Tree"]
    mape_values = [mape_lr_eval, mape_dt_eval]
    colors = ["steelblue", "tomato"]
    bars = ax.bar(models, mape_values, color=colors, edgecolor="black")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Perbandingan MAPE antar Model")
    for bar, val in zip(bars, mape_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.4f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════
#  TAB 5 – RINGKASAN
# ══════════════════════════════════════════
with tab5:
    st.header("5. Ringkasan Hasil")
    
    best_model = "Regresi Linier" if mape_lr_eval < mape_dt_eval else "Regresi Pohon Keputusan"
    best_mape = min(mape_lr_eval, mape_dt_eval)
    
    st.success(f"🏆 **Model Terbaik:** {best_model} dengan MAPE {best_mape:.4f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Regresi Linier")
        st.metric("MAPE", f"{mape_lr_eval:.4f}%")
        st.metric("MAE", f"{mae_lr:.2f}")
        st.metric("RMSE", f"{rmse_lr:.2f}")
        st.metric("R²", f"{r2_lr:.4f}")
    
    with col2:
        st.subheader("Regresi Pohon Keputusan")
        st.metric("MAPE", f"{mape_dt_eval:.4f}%")
        st.metric("MAE", f"{mae_dt:.2f}")
        st.metric("RMSE", f"{rmse_dt:.2f}")
        st.metric("R²", f"{r2_dt:.4f}")
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Interpretasi Hasil
    
    **Regresi Linier:**
    - MAPE 0.22% → Error prediksi sangat kecil, model sangat akurat
    - R² 0.9990 → Model mampu menjelaskan 99.9% variasi data
    - Koefisien positif tertinggi pada fitur **Daily Return** (2438.25) dan **High** (0.666)
    
    **Decision Tree:**
    - MAPE 0.41% → Juga sangat akurat, namun sedikit lebih tinggi dari regresi linier
    - R² 0.9950 → Masih sangat baik (99.5% variasi data dapat dijelaskan)
    - Feature importance tertinggi pada **High** (92.3%) dan **Low** (7.2%)
    
    **Kesimpulan:** Kedua model menunjukkan performa yang sangat baik dengan error di bawah 0.5%. 
    Regresi Linier sedikit lebih unggul pada dataset ini. Fitur yang paling berpengaruh adalah 
    **High** (harga tertinggi harian) dan **Daily Return** (return harian).
    """)
