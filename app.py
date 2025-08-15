import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="MEO 監視＆提案デモ", layout="wide")

# -----------------------------
# データ読み込み（なければ自動生成）
# -----------------------------
@st.cache_data
def load_rankings():
    try:
        df = pd.read_csv("data/rankings.csv", parse_dates=["date"])
    except Exception:
        # 疑似データ自動生成：過去30日×3KW×3ホテル
        dates = pd.date_range(datetime.today() - timedelta(days=29), periods=30)
        keywords = ["なんば ホテル", "大阪 シティホテル", "心斎橋 宿泊"]
        hotels = ["ホテルザグランデ", "ホテルA", "ホテルB"]
        rows = []
        rng = np.random.default_rng(42)
        for kw in keywords:
            base = rng.integers(2, 6)  # グランデの基準
            for h in hotels:
                drift = rng.normal(0, 0.15)  # 競合差分
                for i, d in enumerate(dates):
                    noise = rng.normal(0, 0.7)
                    rank = base + (0 if h == "ホテルザグランデ" else rng.integers(0, 4)) + noise + drift
                    rank = max(1, min(15, round(rank + i*0.02)))  # 1〜15位の範囲
                    rows.append([d.date(), kw, h, rank])
        df = pd.DataFrame(rows, columns=["date", "keyword", "hotel", "rank"])
    return df

@st.cache_data
def load_competitors():
    try:
        df = pd.read_csv("data/competitors.csv")
    except Exception:
        df = pd.DataFrame([
            ["ホテルザグランデ", 85, 420, 4.3, 0, "月1回"],
            ["ホテルA",          92, 358, 4.5, 3, "週2回"],
            ["ホテルB",          88, 290, 4.4, 1, "週1回"],
        ], columns=["hotel","score","reviews","rating","photos_added_this_week","post_freq"])
    return df

rankings = load_rankings()
comps = load_competitors()

# UI ヘッダ
st.title("MEO 監視＆提案ツール（デモ）")
st.caption("※ 擬似データ。実運用時はCSV/スプレッドシート差し替えで即稼働")

# フィルタ
col_f1, col_f2, col_f3 = st.columns([1,1,1])
with col_f1:
    kw = st.selectbox("キーワード", sorted(rankings["keyword"].unique()))
with col_f2:
    days = st.slider("表示日数", min_value=7, max_value=60, value=30, step=1)
with col_f3:
    show_hotels = st.multiselect("ホテル選択", sorted(rankings["hotel"].unique()),
                                 default=sorted(rankings["hotel"].unique()))

cutoff = rankings["date"].max() - pd.to_timedelta(days-1, unit="D")
dfv = rankings[(rankings["keyword"]==kw) & (rankings["hotel"].isin(show_hotels)) & (rankings["date"]>=cutoff)].copy()

# KPI（現時点順位・先週比）
today = dfv["date"].max()
last_week = today - pd.to_timedelta(7, "D")
today_rows = dfv[dfv["date"]==today]
lw_rows = dfv[dfv["date"]==last_week]

c1,c2,c3 = st.columns(3)
def get_rank(h, dframe):
    r = dframe[dframe["hotel"]==h]
    return int(r["rank"].values[0]) if not r.empty else None

my_rank_today = get_rank("ホテルザグランデ", today_rows)
my_rank_lw    = get_rank("ホテルザグランデ", lw_rows)
delta = None
if my_rank_today and my_rank_lw:
    delta = my_rank_lw - my_rank_today  # 上がるほど +（良化）
c1.metric("現在順位（自館）", f"{my_rank_today if my_rank_today else '-'} 位",
          f"{'+' if (delta and delta>=0) else ''}{delta if delta else 0} vs 先週")

# 競合最良順位
comp_best = (
    today_rows[today_rows["hotel"]!="ホテルザグランデ"]
    .sort_values("rank", ascending=True)
    .head(1)
)
best_hotel = comp_best["hotel"].values[0] if not comp_best.empty else "-"
best_rank = int(comp_best["rank"].values[0]) if not comp_best.empty else "-"
c2.metric("競合ベスト順位", f"{best_rank} 位", best_hotel)

# 平均順位（小さいほど良い）
avg_rank = today_rows["rank"].mean() if not today_rows.empty else None
c3.metric("今日の平均順位（選択ホテル）", f"{avg_rank:.1f}" if avg_rank else "-")

st.divider()

# 順位推移グラフ
st.subheader(f"順位推移：{kw}")
fig = px.line(dfv, x="date", y="rank", color="hotel", markers=True)
fig.update_yaxes(autorange="reversed", title="順位（1位が上）", range=[15,1])
fig.update_xaxes(title="日付")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# 競合比較テーブル
st.subheader("競合比較（スコア＆KPI）")
# 軽い色付け
show = comps.copy()
show["優位/要改善"] = np.where(show["hotel"]=="ホテルザグランデ", "★自館", "")
st.dataframe(show.style
             .background_gradient(subset=["score"], cmap="Greens")
             .background_gradient(subset=["reviews"], cmap="Blues")
             .background_gradient(subset=["rating"], cmap="Oranges"),
             use_container_width=True)

st.divider()

# 改善提案（サンプルロジック）
st.subheader("自動改善提案（サンプル）")
suggestions = []

# 1) 順位急落アラート
my_today = my_rank_today
my_yest  = get_rank("ホテルザグランデ", dfv[dfv["date"]==today - pd.to_timedelta(1,"D")])
if my_today and my_yest and (my_today - my_yest) >= 3:
    suggestions.append(f"⚠️ 順位急落：{kw} で {my_yest}位 → {my_today}位（- {my_today - my_yest}）")

# 2) 競合に負けている場合の打ち手
if isinstance(best_rank, int) and my_rank_today and my_rank_today > best_rank:
    # 競合の活動（写真/投稿）を参照
    comp_row = comps[comps["hotel"]!= "ホテルザグランデ"].sort_values("score", ascending=False).head(1).iloc[0]
    if comp_row["photos_added_this_week"] >= 2:
        suggestions.append("🖼️ 写真追加推奨：上位競合が今週2枚以上追加。今週は館内写真を3枚追加。")
    suggestions.append("📝 最新情報の投稿頻度強化：週2回を目標に。")
    suggestions.append(f"🗺️ 説明文微修正：『{kw}』の文言を自然に追記。")

# 3) 防衛ライン（トップ5死守）
if my_rank_today and my_rank_today > 5:
    suggestions.append("🎯 まずはトップ5復帰：写真3枚・最新情報2本・Q&A1件更新を今週中に。")

if not suggestions:
    suggestions = ["✅ 先週比で良好です。今週は現状維持（写真1枚追加のみ）でOK。"]

for s in suggestions:
    st.write("- " + s)

st.caption("※ 提案ロジックはデモ用。実運用では自館データ・競合動向に合わせて調整します。")
