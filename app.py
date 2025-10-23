# app.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import japanize_matplotlib


# ===================== ユーティリティ =====================
def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna().astype(str))
        return True
    except Exception:
        return False

def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str), errors='coerce')

def compute_thresholds(fit_lo: float, fit_hi: float, step: int = 100_000):
    """右→左（降順）。上限=fit_hi -> 下限=fit_lo（step刻み）"""
    fit_lo_i = int(np.floor(fit_lo / step) * step)
    fit_hi_i = int(np.ceil(fit_hi / step) * step)
    if fit_hi_i <= fit_lo_i + step:
        fit_hi_i = fit_lo_i + 2 * step  # 幅が狭すぎる時の安全策
    return list(range(fit_hi_i, fit_lo_i, -step))

# ===================== データ入力 =====================
st.set_page_config(page_title="Fitness 可視化アプリ", layout="wide")

st.title("Fitness 閾値ごとの統計可視化（Streamlit版）")

# A) CSVアップロード or B) 同梱CSVを読む
uploaded = st.sidebar.file_uploader("CSVをアップロード（任意）", type=["csv"])
if uploaded is not None:
    df_x_unique_labels_selected = pd.read_csv(uploaded)
else:
    # 同梱ファイル名を変える場合はここを編集
    try:
        df_x_unique_labels_selected = pd.read_csv("df_x_unique_labels_selected.csv")
    except Exception:
        st.warning("CSVが読み込めていません。左のサイドバーからCSVをアップロードしてください。")
        st.stop()

# 元コード準拠：cluster は文字列扱い
df_all = df_x_unique_labels_selected.copy()
df_all["cluster"] = df_all["cluster"].astype(str)

# 描画対象列（Fitness, cluster は除外）
VALUE_COLS = [c for c in df_all.columns if c not in ["Fitness", "cluster"]]
if not VALUE_COLS:
    st.error("描画対象列が見つかりません（Fitness と cluster 以外の列が必要です）。")
    st.stop()

# Fitness のグローバル最小/最大
FIT_MIN_GLOBAL = float(df_all["Fitness"].astype(float).min())
FIT_MAX_GLOBAL = float(df_all["Fitness"].astype(float).max())

# ===================== サイドバーUI =====================
st.sidebar.header("基本設定")

# cluster 選択（“(すべて)” 追加）
cluster_options = ["(すべて)"] + sorted(df_all["cluster"].unique().tolist())
selected_cluster = st.sidebar.selectbox("cluster", cluster_options, index=0)

# Fitness 範囲（Range Slider）
default_lo = max(8_000_000.0, FIT_MIN_GLOBAL)
default_hi = min(10_100_000.0, FIT_MAX_GLOBAL)
fit_lo, fit_hi = st.sidebar.slider(
    "Fitness範囲",
    min_value=float(FIT_MIN_GLOBAL),
    max_value=float(FIT_MAX_GLOBAL),
    value=(float(default_lo), float(default_hi)),
    step=100_000.0,
    format="%.0f",
)

st.sidebar.markdown("---")
st.sidebar.header("絞り込み設定")

# フィルタ ON/OFF
filter_enable = st.sidebar.radio("フィルタ", ["絞り込みしない", "絞り込みする"], index=0)

# 値候補更新：cluster/Fitness/列に依存
def get_base_df(selected_cluster: str) -> pd.DataFrame:
    if selected_cluster == "(すべて)":
        return df_all.copy()
    return df_all[df_all["cluster"] == selected_cluster].copy()

dfc_for_opts = get_base_df(selected_cluster)
dfc_for_opts = dfc_for_opts[(dfc_for_opts["Fitness"].astype(float) >= fit_lo) & (dfc_for_opts["Fitness"].astype(float) < fit_hi)]

# フィルタ列
filter_col = st.sidebar.selectbox("絞込列", VALUE_COLS, index=0, disabled=(filter_enable == "絞り込みしない"))

# 絞込値の候補（列が数値か文字かで分岐）
if filter_enable == "絞り込みする":
    ser = dfc_for_opts[filter_col]
    if is_numeric_series(ser):
        options_vals = sorted(pd.unique(to_numeric_safe(ser).dropna()))
        # 初期選択は先頭1件
        selected_vals = st.sidebar.multiselect("絞込値（複数可）", options=options_vals, default=options_vals[:1])
    else:
        options_vals = sorted(pd.unique(ser.fillna("NaN").astype(str)))
        selected_vals = st.sidebar.multiselect("絞込値（複数可）", options=options_vals, default=options_vals[:1])
    include_empty = st.sidebar.checkbox("欠損(NaN)も含める", value=False)
else:
    selected_vals = []
    include_empty = False

st.sidebar.markdown("---")
st.sidebar.header("描画設定")
plot_mode = st.sidebar.radio("表示", ["全列", "列を選ぶ"], index=0)
plot_col = st.sidebar.selectbox("描画列", VALUE_COLS, index=0, disabled=(plot_mode == "全列"))
show_quartiles = st.sidebar.checkbox("Q1/Q3も表示", value=False)

# ===================== データフィルタ =====================
def apply_filter(dfc: pd.DataFrame) -> pd.DataFrame:
    # Fitness 範囲は常に適用
    dfc = dfc[(dfc["Fitness"].astype(float) >= fit_lo) & (dfc["Fitness"].astype(float) < fit_hi)]
    if filter_enable == "絞り込みしない":
        return dfc

    ser = dfc[filter_col]
    if len(selected_vals) == 0 and not include_empty:
        return dfc

    if is_numeric_series(ser):
        ser_num = to_numeric_safe(ser)
        mask_val = ser_num.isin(list(selected_vals))
    else:
        ser_str = ser.fillna("NaN").astype(str)
        mask_val = ser_str.isin(set(selected_vals))

    if include_empty:
        mask_val = mask_val | ser.isna()

    return dfc[mask_val]

base_df = get_base_df(selected_cluster)
work_df = apply_filter(base_df)

# ===================== 情報表示 =====================
st.caption(
    f"対象件数: {len(work_df)}｜Fitness範囲: [{fit_lo:,.0f}, {fit_hi:,.0f})｜"
    f"cluster: {selected_cluster}"
)

# プレビュー（折りたたみ）
with st.expander("データプレビュー（先頭50行）", expanded=False):
    st.dataframe(work_df.head(50), use_container_width=True)

# ===================== 描画関数 =====================
def plot_columns_for_df(dfc: pd.DataFrame, columns, show_quartiles=False):
    if dfc.empty:
        st.warning("※ 該当データがありません（条件を緩めてください）。")
        return

    thresholds = compute_thresholds(fit_lo, fit_hi, step=100_000)
    x_right = fit_lo  # 右端=選択したFitness範囲の下限

    for col in columns:
        stats = {"min": [], "q1": [], "median": [], "q3": [], "max": [], "mean": []}
        for th in thresholds:
            vals = dfc.loc[dfc["Fitness"].astype(float) <= th, col]
            vals = to_numeric_safe(vals).dropna()
            if len(vals) > 0:
                stats["min"].append(vals.min())
                stats["q1"].append(vals.quantile(0.25))
                stats["median"].append(vals.median())
                stats["q3"].append(vals.quantile(0.75))
                stats["max"].append(vals.max())
                stats["mean"].append(vals.mean())
            else:
                stats["min"].append(np.nan)
                stats["q1"].append(np.nan)
                stats["median"].append(np.nan)
                stats["q3"].append(np.nan)
                stats["max"].append(np.nan)
                stats["mean"].append(np.nan)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, stats["min"], label="最小値")
        if show_quartiles:
            ax.plot(thresholds, stats["q1"], label="第1四分位数")
        ax.plot(thresholds, stats["median"], label="中央値")
        if show_quartiles:
            ax.plot(thresholds, stats["q3"], label="第3四分位数")
        ax.plot(thresholds, stats["max"], label="最大値")
        ax.plot(thresholds, stats["mean"], label="平均値")
        ax.set_xlabel("Fitness閾値")
        ax.set_ylabel(col)
        ax.set_title(f"Fitness閾値ごとの {col} 統計値 (cluster={selected_cluster})")
        ax.invert_xaxis()
        ax.set_xlim(right=x_right)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

# ===================== 実行（ボタンなし＝即時反映） =====================
if plot_mode == "全列":
    columns = VALUE_COLS
else:
    columns = [plot_col]

plot_columns_for_df(work_df, columns, show_quartiles=show_quartiles)
