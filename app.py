# app.py
# MEO 監視＆提案ツール（1KW × 10ホテル 版）
# - 公開CSV（Googleスプレッドシートの “rankings” をCSV公開）を読み込み
# - キーワードでフィルタ、ホテルを選択
# - KPI（自館の現在順位/競合ベスト/平均）をNaN安全に表示
# - 順位推移チャートと簡易KPIテーブルを表示
# 期待CSV列: date, keyword, hotel, rank

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ==== ページ設定 ============================================================
st.set_page_config(page_title="MEO 監視＆提案ツール（デモ）", layout="wide")

# ==== 自館・監視対象（10ホテル） ===========================================
MY_HOTEL = "ホテルザグランデ"
TARGET_HOTELS = [
    "ホテルザグランデ",                        # 自館
    "ホテルヒラリーズ心斎橋",
    "ドーミーイン PREMIUM なんば",
    "ホテルモントレ グラスミア大阪",
    "KOKO HOTEL 大阪心斎橋",
    "DEL style 大阪心斎橋 by Daiwa Roynet Hotel",
    "ホテルロイヤルクラシック大阪",
    "クロスホテル大阪",
    "ホテルコード心斎橋",                      # ← 追加
    "コンフォートホテル大阪心斎橋",            # ← 追加
]
DEFAULT_KW = "心斎橋 ホテル"

# ==== ユーティリティ ========================================================
@st.cache_data(ttl=60)
def load_csv(url_or_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url_or_path)
        return df
    except Exception:
        # ローカルのデモCSV（無い場合は空DataFrame）
        fallback = "data/rankings.csv"
        if os.path.exists(fallback):
            return pd.read_csv(fallback)
        return pd.DataFrame(columns=["date", "keyword", "hotel", "rank"])

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    # 必須列マッピング（大文字/全角スペースなどを吸収）
    mapping = {}
    for need in ["date", "keyword", "hotel", "rank"]:
        found = None
        for c in df.columns:
            key = c.lower().strip().replace("　", " ")
            if key == need:
                found = c
                break
        if found is None:
            # 代表的な別名候補
            for c in df.columns:
                key = c.lower().strip().replace("　", " ")
                if need == "date" and key in ["日付", "date"]:
                    found = c; break
                if need == "keyword" and key in ["kw", "キーワード", "keyword"]:
                    found = c; break
                if need == "hotel" and key in ["施設", "ホテル", "hotel"]:
                    found = c; break
                if need == "rank" and key in ["順位", "rank"]:
                    found = c; break
        if found:
            mapping[found] = need
    if mapping:
        df = df.rename(columns=mapping)
    # 欠損列を補完
    for need in ["date", "keyword", "hotel", "rank"]:
        if need not in df.columns:
            df[need] = np.nan
    return df[["date", "keyword", "hotel", "rank"]].copy()

def to_datetime_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    # 日付をdatetimeに
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # rankを数値化（NaNはNaNのまま）
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    # ホテル名の前後スペースを正規化
    df["hotel"] = df["hotel"].astype(str).str.strip()
    return df.sort_values(["date", "hotel"]).reset_index(drop=True)

def guard_df(df: pd.DataFrame) -> bool:
    return (not df.empty) and df["date"].notna().any()

# ==== データソース設定（公開CSV URL） ======================================
with st.expander("データソース設定（任意：Googleスプレッドシートの公開CSV URLを貼ると自動読み込み）", expanded=True):
    default_url = st.session_state.get("csv_url", "")
    csv_url = st.text_input(
        "公開CSVのURL（空欄ならローカルCSV。無ければ模擬データ）",
        value=default_url,
        placeholder="https://docs.google.com/spreadsheets/d/e/2PACX-1vRYbCtXeY5ySt_2VqeT5tDsT5nvnYK-3gOyCrvaJUAp1euQ_b3Nx7_p7tnHR91Fa-FkIyLalBlQPT_5/pub?gid=0&single=true&output=csv",
    )
    st.session_state["csv_url"] = csv_url

# 読み込み
raw = load_csv(csv_url) if csv_url else load_csv("data/rankings.csv")
df = normalize_columns(raw)
df = to_datetime_and_sort(df)

if guard_df(df):
    st.success("公開CSVから読み込みました。")
else:
    st.warning("有効なデータが見つかりませんでした（ヘッダ: date, keyword, hotel, rank）。")

# ==== UI（フィルタ） ========================================================
st.title("MEO 監視＆提案ツール（デモ）")

# キーワード（1本運用だがCSVから存在するものを採用）
kw_list = sorted([k for k in df["keyword"].dropna().unique().tolist() if str(k).strip() != ""])
if DEFAULT_KW in kw_list:
    current_kw = DEFAULT_KW
elif kw_list:
    current_kw = kw_list[0]
else:
    # データが空なら既定KW
    current_kw = DEFAULT_KW

col_kw, col_days = st.columns([1,1])
with col_kw:
    kw = st.selectbox("キーワード", options=kw_list if kw_list else [current_kw], index=(kw_list.index(current_kw) if current_kw in kw_list else 0))
with col_days:
    days = st.slider("表示日数", min_value=7, max_value=90, value=30)

# ホテル選択（CSVに存在するものだけ初期選択）
exist_hotels = sorted(df["hotel"].dropna().unique().tolist())
default_hotels = [h for h in TARGET_HOTELS if h in exist_hotels] or exist_hotels
selected_hotels = st.multiselect("ホテル選択", options=exist_hotels, default=default_hotels)

# ==== データ絞り込み ========================================================
cutoff = df["date"].max() - pd.to_timedelta(days-1, "D") if guard_df(df) else None
view = df[(df["keyword"] == kw) & (df["hotel"].isin(selected_hotels))].copy()
if cutoff is not None:
    view = view[view["date"] >= cutoff]

# ==== KPI（NaN安全） ========================================================
def safe_int(v):
    return int(v) if pd.notna(v) else None

def get_rank(hotel_name: str, frame: pd.DataFrame, on_date: pd.Timestamp):
    r = frame[(frame["hotel"] == hotel_name) & (frame["date"] == on_date)]
    if r.empty:
        return None
    return safe_int(r["rank"].values[0])

if guard_df(view) and not view.empty:
    today = view["date"].max()
    last_week = today - pd.to_timedelta(7, "D")
    today_rows = view[view["date"] == today].copy()
    lw_rows    = view[view["date"] == last_week].copy()

    c1, c2, c3 = st.columns(3)

    # 自館
    my_rank_today = get_rank(MY_HOTEL, view, today)
    my_rank_lw    = get_rank(MY_HOTEL, view, last_week)
    delta = None
    if (my_rank_today is not None) and (my_rank_lw is not None):
        delta = my_rank_lw - my_rank_today  # 改善はプラス

    c1.metric(
        "現在順位（自館）",
        f"{my_rank_today if my_rank_today is not None else '-'} 位",
        f"{'+' if (delta is not None and delta >= 0) else ''}{delta if delta is not None else 0} vs 先週",
    )

    # 競合ベスト
    comp_pool = today_rows[(today_rows["hotel"] != MY_HOTEL) & (pd.notna(today_rows["rank"]))].copy()
    if comp_pool.empty:
        best_hotel, best_rank = "-", "-"
    else:
        comp_best = comp_pool.nsmallest(1, "rank")
        best_hotel = comp_best["hotel"].values[0]
        best_rank_val = comp_best["rank"].values[0]
        best_rank = safe_int(best_rank_val) if best_rank_val is not None else "-"
    c2.metric("競合ベスト順位", f"{best_rank} 位" if best_rank != "-" else "-", best_hotel)

    # 平均（NaN除外）
    avg_rank = today_rows["rank"].dropna().mean()
    c3.metric("今日の平均順位（選択ホテル）", f"{avg_rank:.1f}" if pd.notna(avg_rank) else "–")

    if today_rows["rank"].isna().any():
        st.caption("ℹ️ 一部ホテルの rank が未入力です（平均や競合ベストに反映されません）。")
else:
    st.info("この条件でのデータがありません。rankの入力や日数を確認してください。")

# ==== 順位推移（Plotly） ====================================================
st.subheader(f"順位推移：{kw}")
if not view.empty:
    chart_df = view.copy()
    chart_df["rank"] = pd.to_numeric(chart_df["rank"], errors="coerce")
    fig = px.line(
        chart_df,
        x="date",
        y="rank",
        color="hotel",
        markers=True,
        title=None,
    )
    # 順位は小さいほど上なのでY軸を反転
    fig.update_yaxes(autorange="reversed", title_text="順位（上位が上）")
    fig.update_xaxes(title_text="日付")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("データなし")

# ==== 簡易 競合比較（スコア＆KPIテーブル：ダミー or 外部CSV） ===============
# ここはデモ用。実運用では別CSV/スプレッドシートに載せればOK。
def load_competitors() -> pd.DataFrame:
    path = "data/competitors.csv"
    if os.path.exists(path):
        try:
            dfc = pd.read_csv(path)
            return dfc
        except Exception:
            pass
    # サンプル（3行）
    return pd.DataFrame({
        "hotel": ["ホテルザグランデ", "ホテルA", "ホテルB"],
        "score": [85, 92, 88],
        "reviews": [420, 358, 290],
        "rating": [4.3, 4.5, 4.4],
        "photos_added_this_week": [0, 3, 1],
        "post_freq": ["月1回", "週2回", "週1回"],
        "優位/要改善": ["★自館", "", ""],
    })

st.subheader("競合比較（スコア＆KPI）")
comp = load_competitors()
if not comp.empty:
    try:
        st.dataframe(
            comp.style
            .background_gradient(subset=["score"], cmap="Greens")
            .background_gradient(subset=["reviews"], cmap="Blues")
            .background_gradient(subset=["rating"], cmap="Oranges"),
            use_container_width=True
        )
    except Exception:
        st.dataframe(comp, use_container_width=True)
else:
    st.write("（KPIデータ未設定）")

# ==== 末尾メモ ==============================================================
st.caption("※ rankは手動入力です。運用は『1KW×10ホテル（週10件入力）』に最適化。未入力があっても落ちません。")
