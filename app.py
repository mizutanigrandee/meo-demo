# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# --------------------------------------------------------------------------------------
# ページ設定
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="MEO 監視＆提案ツール（デモ）", layout="wide")

# --------------------------------------------------------------------------------------
# 固定：商戦向けキーワード（5つ）
# --------------------------------------------------------------------------------------
KEYWORDS_FIXED = [
    "心斎橋 ホテル",
    "なんば ホテル",
    "心斎橋 ビジネスホテル",
    "難波 ビジネスホテル",
    "大阪 難波 ホテル",
]

# --------------------------------------------------------------------------------------
# 固定：競合ホテル（7軒）
# ※ CSV側の表記と揃うほど初期選択がキレイに効きます
# --------------------------------------------------------------------------------------
COMPETITOR_HOTELS_FIXED = [
    "ホテルヒラリーズ心斎橋",
    "ドーミーイン PREMIUM なんば",
    "ホテルモントレ グラスミア大阪",
    "KOKO HOTEL 大阪心斎橋",
    "DEL style 大阪心斎橋 by Daiwa Roynet Hotel",
    "ホテルロイヤルクラシック大阪",
    "クロスホテル大阪",
]

# --------------------------------------------------------------------------------------
# データ読み込み（CSV or 擬似データ）
# --------------------------------------------------------------------------------------
@st.cache_data
def load_rankings_from_local():
    """data/rankings.csv を読む。無ければ None を返す"""
    try:
        df = pd.read_csv("data/rankings.csv")
        return df
    except Exception:
        return None

@st.cache_data
def load_competitors_from_local():
    """data/competitors.csv を読む。無ければデフォルトDataFrameを返す"""
    try:
        df = pd.read_csv("data/competitors.csv")
        return df
    except Exception:
        df = pd.DataFrame([
            ["ホテルザグランデ", 85, 420, 4.3, 0, "月1回"],
            ["ホテルヒラリーズ心斎橋", 90, 650, 4.3, 2, "週1回"],
            ["ドーミーイン PREMIUM なんば", 93, 1200, 4.5, 3, "週2回"],
            ["ホテルモントレ グラスミア大阪", 91, 2100, 4.4, 1, "週1回"],
            ["KOKO HOTEL 大阪心斎橋", 88, 900, 4.2, 1, "週1回"],
            ["DEL style 大阪心斎橋 by Daiwa Roynet Hotel", 89, 800, 4.2, 1, "週1回"],
            ["ホテルロイヤルクラシック大阪", 95, 2500, 4.6, 2, "週1回"],
            ["クロスホテル大阪", 94, 3000, 4.5, 2, "週2回"],
        ], columns=["hotel","score","reviews","rating","photos_added_this_week","post_freq"])
        return df

@st.cache_data
def generate_synthetic_rankings(days=30, seed=42):
    """擬似データ生成：過去days日 x 固定KW x （自館＋競合）"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(datetime.today().date() - timedelta(days=days-1), periods=days, freq="D")
    keywords = KEYWORDS_FIXED
    hotels = (["ホテルザグランデ"] + COMPETITOR_HOTELS_FIXED)[:8]  # 自館＋主要競合（最大8軸程度）
    rows = []
    for kw in keywords:
        base_rank = rng.integers(2, 6)  # 自館の基準位置
        for h in hotels:
            hotel_offset = 0 if h == "ホテルザグランデ" else rng.integers(0, 5)
            drift = rng.normal(0, 0.05)
            cur = base_rank + hotel_offset + drift
            for i, d in enumerate(dates):
                noise = rng.normal(0, 0.6)
                rank = cur + noise + (i * 0.02 if h != "ホテルザグランデ" else 0)
                rank = int(max(1, min(15, round(rank))))
                rows.append([d, kw, h, rank])
    df = pd.DataFrame(rows, columns=["date", "keyword", "hotel", "rank"])
    return df

@st.cache_data
def load_rankings_from_url(csv_url: str):
    """公開CSV（Googleスプレッドシート等）から読む"""
    df = pd.read_csv(csv_url)
    return df

def to_datetime_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """date列をdatetime64化し、NaT除去→昇順ソート"""
    if "date" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df

def uniq_sorted_str(series: pd.Series):
    """NaN除去→文字列化→ユニーク→ソート"""
    return sorted(series.dropna().astype(str).unique().tolist())

# --------------------------------------------------------------------------------------
# サイド/上部：データソース切替（任意）
# --------------------------------------------------------------------------------------
with st.expander("データソース設定（任意：Googleスプレッドシートの公開CSV URLを貼ると自動読込）", expanded=False):
    csv_url_input = st.text_input(
        "公開CSVのURL（空欄ならローカルCSV→無ければ擬似データ）",
        value="",
        placeholder="https://docs.google.com/spreadsheets/d/.../pub?output=csv",
    )

# ランキングデータの確定（優先順位：URL > ローカルCSV > 擬似）
rankings_raw = None
if csv_url_input.strip():
    try:
        rankings_raw = load_rankings_from_url(csv_url_input.strip())
        st.success("公開CSVから読み込みました。")
    except Exception as e:
        st.warning(f"公開CSVの読込に失敗：{e}。ローカルCSV/擬似データへフォールバックします。")

if rankings_raw is None:
    rankings_raw = load_rankings_from_local()
    if rankings_raw is not None:
        st.info("data/rankings.csv を読み込みました。")

if rankings_raw is None:
    rankings_raw = generate_synthetic_rankings()
    st.info("擬似データで表示しています。公開CSVまたは data/rankings.csv を用意すると置き換わります。")

# 前処理：date型統一
rankings = to_datetime_and_sort(rankings_raw)

# 競合KPI（ローカル or デフォルト）
comps = load_competitors_from_local()

# --------------------------------------------------------------------------------------
# UI：フィルタ（KW・期間・ホテル）
# --------------------------------------------------------------------------------------
st.title("MEO 監視＆提案ツール（デモ）")

col_f1, col_f2, col_f3 = st.columns([1,1,2])

with col_f1:
    data_kws = uniq_sorted_str(rankings["keyword"]) if "keyword" in rankings.columns else []
    # 表示候補：固定KW + データ上のKW（重複除去）
    kw_options = list(dict.fromkeys(KEYWORDS_FIXED + [k for k in data_kws if k not in KEYWORDS_FIXED]))
    # 既定選択：固定KWのうち「データにあるもの」→ なければ data_kws 先頭 → 固定1つ目
    default_kw = next((k for k in KEYWORDS_FIXED if k in data_kws), (data_kws[0] if data_kws else KEYWORDS_FIXED[0]))
    kw = st.selectbox("キーワード", kw_options, index=kw_options.index(default_kw))

with col_f2:
    days = st.slider("表示日数", min_value=7, max_value=90, value=30, step=1)

with col_f3:
    data_hotels = uniq_sorted_str(rankings["hotel"]) if "hotel" in rankings.columns else []
    # 表示候補：データのホテル + 自館 + 固定競合 を統合（重複除去）
    hotels_all = list(dict.fromkeys(data_hotels + ["ホテルザグランデ"] + COMPETITOR_HOTELS_FIXED))
    # 既定選択：自館 + 固定競合（リストにあるものだけ）
    default_hotels = ["ホテルザグランデ"] + COMPETITOR_HOTELS_FIXED
    show_hotels = st.multiselect("ホテル選択", hotels_all, default=[h for h in default_hotels if h in hotels_all])

    # データに無いホテルが選択された場合の注意書き
    missing = [h for h in show_hotels if h not in data_hotels]
    if missing:
        st.caption("⚠️ これらのホテルはrankingsデータに未登録です: " + " / ".join(missing))


# フィルタ適用
if "date" not in rankings.columns:
    st.error("データに 'date' 列がありません。CSVのヘッダを確認してください。")
    st.stop()

cutoff = rankings["date"].max() - pd.to_timedelta(days-1, unit="D")
dfv = rankings[
    (rankings["keyword"].astype(str) == str(kw)) &
    (rankings["hotel"].isin(show_hotels)) &
    (rankings["date"] >= cutoff)
].copy()

# 空データのガード
if dfv.empty:
    st.info("選択条件に一致するデータがありません。CSV/スプレッドシート、フィルタ条件をご確認ください。")
    st.stop()

# --------------------------------------------------------------------------------------
# KPI（現時点順位・先週比など）
# --------------------------------------------------------------------------------------
today = dfv["date"].max()
last_week = today - pd.to_timedelta(7, "D")
today_rows = dfv[dfv["date"] == today]
lw_rows = dfv[dfv["date"] == last_week]

def get_rank(hotel_name: str, frame: pd.DataFrame):
    r = frame[frame["hotel"] == hotel_name]
    return int(r["rank"].values[0]) if not r.empty else None

c1, c2, c3 = st.columns(3)

my_rank_today = get_rank("ホテルザグランデ", today_rows)
my_rank_lw    = get_rank("ホテルザグランデ", lw_rows)
delta = None
if (my_rank_today is not None) and (my_rank_lw is not None):
    delta = my_rank_lw - my_rank_today  # 良化で+、悪化で-

c1.metric("現在順位（自館）", f"{my_rank_today if my_rank_today is not None else '-'} 位",
          f"{'+' if (delta is not None and delta >= 0) else ''}{delta if delta is not None else 0} vs 先週")

comp_best = (
    today_rows[today_rows["hotel"] != "ホテルザグランデ"]
    .sort_values("rank", ascending=True)
    .head(1)
)
best_hotel = comp_best["hotel"].values[0] if not comp_best.empty else "-"
best_rank = int(comp_best["rank"].values[0]) if not comp_best.empty else "-"
c2.metric("競合ベスト順位", f"{best_rank} 位", best_hotel)

avg_rank = today_rows["rank"].mean() if not today_rows.empty else None
c3.metric("今日の平均順位（選択ホテル）", f"{avg_rank:.1f}" if avg_rank is not None else "-")

st.divider()

# --------------------------------------------------------------------------------------
# グラフ：順位推移
# --------------------------------------------------------------------------------------
st.subheader(f"順位推移：{kw}")
fig = px.line(dfv, x="date", y="rank", color="hotel", markers=True)
fig.update_yaxes(autorange="reversed", title="順位（1位が上）", range=[max(15, dfv['rank'].max()+1), 1])
fig.update_xaxes(title="日付")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------------------
# 競合比較テーブル：色付け（matplotlibが必要）
# --------------------------------------------------------------------------------------
st.subheader("競合比較（スコア＆KPI）")
show = comps.copy()
# 自館行が無ければ追加（スコアは仮）
if "ホテルザグランデ" not in set(show["hotel"]):
    show = pd.concat([
        pd.DataFrame([["ホテルザグランデ", 85, 420, 4.3, 0, "月1回"]],
                     columns=show.columns),
        show
    ], ignore_index=True)

show["優位/要改善"] = np.where(show["hotel"] == "ホテルザグランデ", "★自館", "")

try:
    styled = (show.style
              .background_gradient(subset=["score"], cmap="Greens")
              .background_gradient(subset=["reviews"], cmap="Blues")
              .background_gradient(subset=["rating"], cmap="Oranges"))
    st.dataframe(styled, use_container_width=True)
except Exception:
    # 万一matplotlib未導入でも表示は継続
    st.dataframe(show, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------------------
# 改善提案（シンプルなサンプルロジック）
# --------------------------------------------------------------------------------------
st.subheader("自動改善提案（サンプル）")
suggestions = []

# 1) 急落アラート（当日 vs 前日）
yest = today - pd.to_timedelta(1, "D")
my_yest = get_rank("ホテルザグランデ", dfv[dfv["date"] == yest])
if (my_rank_today is not None) and (my_yest is not None) and (my_rank_today - my_yest >= 3):
    suggestions.append(f"⚠️ 順位急落：{kw} で {my_yest}位 → {my_rank_today}位（-{my_rank_today - my_yest}）")

# 2) 競合ベストに劣後 → 打ち手
if isinstance(best_rank, int) and (my_rank_today is not None) and (my_rank_today > best_rank):
    # 競合の活動量をざっくり参照（もっともスコアが高い競合）
    comp_row = show[show["hotel"] != "ホテルザグランデ"].sort_values("score", ascending=False).head(1)
    if not comp_row.empty:
        photos_add = int(comp_row["photos_added_this_week"].values[0]) if "photos_added_this_week" in comp_row else 0
        if photos_add >= 2:
            suggestions.append("🖼️ 写真追加推奨：上位競合が今週2枚以上追加。今週は館内写真を3枚追加。")
    suggestions.append("📝 最新情報の投稿頻度強化：週2回を目標。")
    suggestions.append(f"🗺️ 説明文微修正：『{kw}』を自然に含める。")

# 3) トップ5防衛
if (my_rank_today is not None) and (my_rank_today > 5):
    suggestions.append("🎯 トップ5復帰：写真3枚・最新情報2本・Q&A1件を今週中に。")

if not suggestions:
    suggestions = ["✅ 先週比で良好。今週は現状維持（写真1枚追加のみ）でOK。"]

for s in suggestions:
    st.write("- " + s)

st.caption("※ ロジックはデモ用。自館/競合の実データに合わせて閾値・提案内容はチューニングします。")
