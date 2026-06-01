"""
フード発注管理アプリ
- Square CSVルートB（2枚同時アップロード）で日次データ取り込み
- 手入力での追記・修正
- ダッシュボード・発注予測
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from square_csv import (
    DayImport,
    load_uploaded_dataframe,
    parse_dual_csv_upload,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
APP_TITLE = "Square CSV連携 フード発注管理"
DAILY_REVENUE_TARGET = 1_000_000
COLD_STORAGE_CAPACITY = 300
DATA_VERSION = 2  # 仕様変更時に上げると data/ を初期化

DATA_DIR = Path(__file__).resolve().parent / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
DAILY_CSV = DATA_DIR / "daily_sales.csv"
VERSION_FILE = DATA_DIR / ".data_version"

DEFAULT_PRODUCTS: list[dict[str, Any]] = [
    {"name": "ハニー＆シトラス", "unit_price": 800},
    {"name": "フルーツ＆ナッツ", "unit_price": 800},
    {"name": "羊羹", "unit_price": 400},
    {"name": "羊羹 みかん", "unit_price": 450},
    {"name": "羊羹 テリーヌ", "unit_price": 500},
    {"name": "羊羹 抹茶", "unit_price": 450},
    {"name": "ワッフル", "unit_price": 900},
    {"name": "パフェ", "unit_price": 1250},
    {"name": "チーズケーキ", "unit_price": 780},
]

EMPTY_DATA_MESSAGE = (
    "データがありません。「Square売上CSVアップロード」または「日次データ入力」から登録してください。"
)

DAILY_SALES_COLUMNS = [
    "date",
    "total_sales",
    "total_customers",
    "product_id",
    "product_name",
    "unit_price",
    "units_sold",
    "created_at",
]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 45%, #0f3460 100%);
            color: #fff; border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1rem;
        }
        .main-title h1 { margin: 0; font-size: 1.6rem; }
        .main-title p { margin: .35rem 0 0; opacity: .88; }
        div[data-testid="stMetric"] {
            border: 1px solid #e8ecf4; border-radius: 10px; padding: .5rem .8rem; background: #f8f9fc;
        }
        .section-title {
            font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 .8rem;
            border-left: 4px solid #e94560; padding-left: .6rem;
        }
        .empty-state-box {
            background: #fff8e6; border: 1px solid #f0d78c; border-radius: 10px;
            padding: 1rem 1.2rem; margin-bottom: 1rem; color: #5a4a1a;
        }
        .empty-state-box strong { color: #1a1a2e; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def is_daily_data_empty(daily_df: pd.DataFrame | None = None) -> bool:
    if daily_df is None:
        daily_df = load_daily_sales()
    return daily_df.empty


def show_empty_data_notice() -> None:
    st.markdown(
        f"""
        <div class="empty-state-box">
            <strong>📂 {EMPTY_DATA_MESSAGE}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# データ永続化
# ---------------------------------------------------------------------------
def init_empty_daily_csv() -> None:
    pd.DataFrame(columns=DAILY_SALES_COLUMNS).to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")


def write_default_products() -> None:
    now = datetime.now().isoformat()
    rows = [
        {
            "product_id": f"FOOD_{idx:03d}",
            "name": p["name"],
            "unit_price": int(p["unit_price"]),
            "is_active": 1,
            "created_at": now,
        }
        for idx, p in enumerate(DEFAULT_PRODUCTS, start=1)
    ]
    pd.DataFrame(rows).to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def reset_application_data() -> None:
    """営業データと商品マスタを初期状態に戻す。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_default_products()
    init_empty_daily_csv()
    VERSION_FILE.write_text(str(DATA_VERSION), encoding="utf-8")


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stored_version = 0
    if VERSION_FILE.exists():
        try:
            stored_version = int(VERSION_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            stored_version = 0

    if stored_version < DATA_VERSION:
        reset_application_data()
        return

    if not PRODUCTS_CSV.exists():
        write_default_products()
    if not DAILY_CSV.exists():
        init_empty_daily_csv()


def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int)
    return df


def active_products(products_df: pd.DataFrame) -> pd.DataFrame:
    active = products_df[products_df["is_active"] == 1].copy()
    return active if not active.empty else products_df.copy()


def load_daily_sales() -> pd.DataFrame:
    if not DAILY_CSV.exists() or DAILY_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(DAILY_CSV, encoding="utf-8-sig")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    for col in ["total_sales", "total_customers", "unit_price", "units_sold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def save_daily_input(
    target_date: date,
    total_sales: int,
    total_customers: int,
    units_by_product: dict[str, int],
    products_df: pd.DataFrame,
) -> None:
    daily_df = load_daily_sales()
    target_ts = to_ts(target_date)
    now = datetime.now().isoformat()

    if not daily_df.empty:
        daily_df = daily_df[daily_df["date"] != target_ts]

    rows = []
    for _, row in products_df.iterrows():
        name = str(row["name"])
        rows.append(
            {
                "date": target_ts.strftime("%Y-%m-%d"),
                "total_sales": int(total_sales),
                "total_customers": int(total_customers),
                "product_id": row["product_id"],
                "product_name": name,
                "unit_price": int(row["unit_price"]),
                "units_sold": int(units_by_product.get(name, 0)),
                "created_at": now,
            }
        )

    merged = pd.concat([daily_df, pd.DataFrame(rows)], ignore_index=True)
    merged.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")


def delete_daily_record(target_date: date) -> None:
    daily_df = load_daily_sales()
    if daily_df.empty:
        return
    daily_df = daily_df[daily_df["date"] != to_ts(target_date)]
    if daily_df.empty:
        init_empty_daily_csv()
    else:
        daily_df.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")


def get_daily_record_by_date(daily_df: pd.DataFrame, target_date: date) -> tuple[dict[str, int], dict[str, int]]:
    if daily_df.empty:
        return {"total_sales": 0, "total_customers": 0}, {}
    target_ts = to_ts(target_date)
    day_rows = daily_df[daily_df["date"] == target_ts]
    if day_rows.empty:
        return {"total_sales": 0, "total_customers": 0}, {}
    totals = {
        "total_sales": int(day_rows["total_sales"].iloc[0]),
        "total_customers": int(day_rows["total_customers"].iloc[0]),
    }
    units_map = dict(zip(day_rows["product_name"].astype(str), day_rows["units_sold"].astype(int)))
    return totals, units_map


def make_daily_summary_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["日付", "店舗総売上", "総客数", "フード販売合計"])
    summary = (
        daily_df.groupby("date", as_index=False)
        .agg(
            total_sales=("total_sales", "first"),
            total_customers=("total_customers", "first"),
            total_food_units=("units_sold", "sum"),
        )
        .sort_values("date", ascending=False)
    )
    summary["date"] = summary["date"].dt.date.astype(str)
    return summary.rename(
        columns={
            "date": "日付",
            "total_sales": "店舗総売上",
            "total_customers": "総客数",
            "total_food_units": "フード販売合計",
        }
    )


def list_recorded_dates(daily_df: pd.DataFrame) -> list[date]:
    if daily_df.empty:
        return []
    return sorted(daily_df["date"].dt.date.unique(), reverse=True)


def bulk_import_days(imports: list[DayImport], products_df: pd.DataFrame, overwrite: bool = True) -> tuple[int, int]:
    """CSV結合結果を daily_sales.csv に保存（日付単位で上書き）。"""
    daily_df = load_daily_sales()
    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    created, updated = 0, 0
    product_list = active_products(products_df)

    for day_data in imports:
        if day_data.day in existing_dates:
            if not overwrite:
                continue
            updated += 1
        else:
            created += 1
        units = {name: day_data.units_by_product.get(name, 0) for name in product_list["name"].astype(str)}
        save_daily_input(day_data.day, day_data.total_sales, day_data.total_customers, units, product_list)

    return created, updated


# ---------------------------------------------------------------------------
# 分析・グラフ
# ---------------------------------------------------------------------------
def get_day_totals(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "total_sales", "total_customers"])
    return (
        daily_df.sort_values("date")
        .groupby("date", as_index=False)[["total_sales", "total_customers"]]
        .first()
    )


def get_latest_product_sales_date(daily_df: pd.DataFrame, product_id: str) -> date | None:
    sub = daily_df[(daily_df["product_id"] == product_id) & (daily_df["units_sold"] > 0)]
    if sub.empty:
        sub = daily_df[daily_df["product_id"] == product_id]
    if sub.empty:
        return None
    return sub["date"].max().date()


def get_latest_business_date(daily_df: pd.DataFrame, day_totals: pd.DataFrame) -> date | None:
    if not day_totals.empty:
        active_days = day_totals[
            (day_totals["total_sales"] > 0) | (day_totals["total_customers"] > 0)
        ]
        if not active_days.empty:
            return active_days["date"].max().date()
        return day_totals["date"].max().date()
    if daily_df.empty:
        return None
    sold = daily_df[daily_df["units_sold"] > 0]
    if not sold.empty:
        return sold["date"].max().date()
    return daily_df["date"].max().date()


def last_week_same_day_sales(daily_df: pd.DataFrame, product_id: str, ref: date) -> int:
    row = daily_df[(daily_df["date"] == to_ts(ref - timedelta(days=7))) & (daily_df["product_id"] == product_id)]
    return int(row["units_sold"].iloc[0]) if not row.empty else 0


def four_week_same_weekday_avg(daily_df: pd.DataFrame, product_id: str, ref: date) -> float:
    ref_ts = to_ts(ref)
    mask = (
        (daily_df["product_id"] == product_id)
        & (daily_df["date"].dt.weekday == ref_ts.weekday())
        & (daily_df["date"] < ref_ts)
    )
    hist = daily_df.loc[mask]
    last_4 = hist[hist["date"] >= to_ts(ref - timedelta(days=28))]
    return float(last_4["units_sold"].mean()) if not last_4.empty else 0.0


def calc_food_selection_rate(daily_df: pd.DataFrame, product_id: str, day_totals: pd.DataFrame, days: int = 7) -> float:
    if daily_df.empty or day_totals.empty:
        return 0.0
    end_date = get_latest_business_date(daily_df, day_totals) or date.today()
    end = to_ts(end_date)
    start = to_ts(end_date - timedelta(days=days))
    sub = daily_df[(daily_df["product_id"] == product_id) & (daily_df["date"] >= start) & (daily_df["date"] <= end)]
    totals = day_totals[(day_totals["date"] >= start) & (day_totals["date"] <= end)]
    customer_sum = int(totals["total_customers"].sum())
    if sub.empty or customer_sum == 0:
        return 0.0
    return float(sub["units_sold"].sum() / customer_sum)


def plot_weekday_comparison(last_week: int, four_week_avg: float, product_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["前週同曜日", "過去4週同曜日平均"],
            y=[last_week, round(four_week_avg, 1)],
            marker_color=["#e94560", "#0f3460"],
            text=[f"{last_week:,} 個", f"{four_week_avg:,.1f} 個"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{product_name} — 同曜日販売数の比較",
        yaxis_title="販売数（個）",
        template="plotly_white",
        height=360,
        showlegend=False,
    )
    return fig


def plot_daily_trend(product_df: pd.DataFrame, product_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=product_df["date"],
            y=product_df["units_sold"],
            mode="lines+markers",
            name="販売個数",
            line=dict(color="#e94560", width=3),
        )
    )
    fig.update_layout(
        title=f"{product_name} — 日次販売推移",
        xaxis_title="日付",
        yaxis_title="販売数（個）",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# タブUI
# ---------------------------------------------------------------------------
def render_square_upload_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">Square売上CSVアップロード（ルートB）</p>', unsafe_allow_html=True)
    st.caption(
        "2枚のCSVを同時にアップロードしてください。"
        " ①商品別マトリックス（日付=横・商品名=縦） ②売上サマリー（日別の総客数・店舗総売上）"
    )

    uploaded_list = st.file_uploader(
        "Square CSVファイル（最大2枚）",
        type=["csv"],
        accept_multiple_files=True,
    )
    overwrite = st.checkbox("既存日付のデータは上書きする", value=True)

    if not uploaded_list:
        if is_daily_data_empty(daily_df):
            st.info(EMPTY_DATA_MESSAGE)
            st.markdown(
                "1. **商品別**の横持ちCSV（9フード品目の日別販売数・売上）  \n"
                "2. **売上サマリー**のCSV（日別の総客数・店舗総売上）  \n"
                "3. 2枚をまとめてドラッグ＆ドロップ → プレビュー確認 → 「データを一括取り込み」"
            )
        else:
            st.info("CSVをアップロードすると、取り込みプレビューが表示されます。")
        return

    files = uploaded_list if isinstance(uploaded_list, list) else [uploaded_list]
    if len(files) > 2:
        st.error("アップロードは2ファイルまでにしてください。")
        return

    st.markdown("#### アップロードされたファイル")
    for f in files:
        kind, _df = load_uploaded_dataframe(f, products_df)
        label = {
            "product_matrix": "商品別マトリックス",
            "sales_summary": "売上サマリー",
            "unknown": "未判定",
        }.get(kind, kind)
        st.write(f"- **{f.name}** → {label}")

    try:
        imports, warnings = parse_dual_csv_upload(files, products_df)
    except Exception as exc:
        st.error(f"CSVの読み込みに失敗しました: {exc}")
        return

    if not imports:
        st.warning("取り込み可能な日次データがありませんでした。")
        return

    preview_rows = [
        {
            "日付": d.day.strftime("%Y-%m-%d"),
            "店舗総売上": d.total_sales,
            "総客数": d.total_customers,
            "フード販売合計": sum(d.units_by_product.values()),
        }
        for d in imports
    ]
    st.markdown("#### 取り込みプレビュー")
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    overlap = [d.day for d in imports if d.day in existing_dates]
    if overlap:
        st.warning(f"既存データと重複する日付: {len(overlap)}日（上書き: {'ON' if overwrite else 'OFF'}）")

    for msg in warnings[:8]:
        st.caption(f"⚠ {msg}")

    if st.button("データを一括取り込み", type="primary", use_container_width=True):
        created, updated = bulk_import_days(imports, products_df, overwrite=overwrite)
        st.success(f"取り込み完了: 新規 {created} 日 / 上書き {updated} 日")
        st.rerun()


def render_daily_input_form(
    products_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    form_key: str,
    default_date: date,
    title: str,
    show_delete: bool = False,
) -> None:
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
    product_list = active_products(products_df)
    product_names = product_list["name"].astype(str).tolist()

    totals, units_map = get_daily_record_by_date(daily_df, default_date)
    has_existing = bool(units_map) or totals["total_sales"] > 0 or totals["total_customers"] > 0

    if has_existing:
        st.info(f"**{default_date.strftime('%Y年%m月%d日')}** の登録データを読み込みました。保存すると上書きされます。")

    with st.form(form_key):
        target_date = st.date_input("日付", value=default_date, key=f"{form_key}_date")

        c1, c2 = st.columns(2)
        with c1:
            total_sales = st.number_input(
                "店舗総売上（円）",
                min_value=0,
                value=int(totals["total_sales"]),
                step=1000,
                key=f"{form_key}_sales",
            )
        with c2:
            total_customers = st.number_input(
                "総客数（人）",
                min_value=0,
                value=int(totals["total_customers"]),
                step=10,
                key=f"{form_key}_customers",
            )

        st.markdown("**フード商品の販売個数**")
        units_input: dict[str, int] = {}
        cols_per_row = 3
        for row_start in range(0, len(product_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, name in enumerate(product_names[row_start : row_start + cols_per_row]):
                with cols[col_idx]:
                    units_input[name] = st.number_input(
                        name,
                        min_value=0,
                        value=int(units_map.get(name, 0)),
                        step=1,
                        key=f"{form_key}_units_{name}",
                    )

        submitted = st.form_submit_button("保存する", type="primary", use_container_width=True)

    if submitted:
        save_daily_input(
            target_date,
            int(total_sales),
            int(total_customers),
            units_input,
            product_list,
        )
        st.success(f"{target_date.strftime('%Y年%m月%d日')} のデータを保存しました。")
        st.rerun()

    if show_delete and has_existing:
        if st.button("この日のデータを削除", type="secondary", key=f"{form_key}_delete"):
            delete_daily_record(default_date)
            st.success(f"{default_date.strftime('%Y年%m月%d日')} のデータを削除しました。")
            st.rerun()


def render_daily_input_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.caption("iPadなどから毎日の売上・客数・9品目の販売数を入力してください。同じ日付で保存すると上書きされます。")
    render_daily_input_form(
        products_df,
        daily_df,
        form_key="main_daily_input",
        default_date=date.today(),
        title="日次データ入力",
    )

    st.markdown('<p class="section-title">登録済みデータ一覧</p>', unsafe_allow_html=True)
    summary = make_daily_summary_table(daily_df)
    if summary.empty:
        st.info("まだ登録された日次データはありません。")
    else:
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_history_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">過去データの確認・修正</p>', unsafe_allow_html=True)

    dates = list_recorded_dates(daily_df)
    if not dates:
        st.info("修正できる過去データがありません。先に日次データを入力してください。")
        return

    selected = st.selectbox(
        "修正する日付",
        dates,
        format_func=lambda d: d.strftime("%Y年%m月%d日"),
        key="history_edit_date",
    )
    render_daily_input_form(
        products_df,
        daily_df,
        form_key="history_edit_form",
        default_date=selected,
        title=f"{selected.strftime('%Y年%m月%d日')} のデータ修正",
        show_delete=True,
    )


def render_dashboard_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">ダッシュボード・可視化</p>', unsafe_allow_html=True)
    if daily_df.empty:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    daily_df = daily_df.sort_values("date")
    day_totals = get_day_totals(daily_df)
    product_list = active_products(products_df)

    selected_name = st.selectbox("対象フード商品", product_list["name"].tolist(), key="dashboard_product")
    selected = product_list[product_list["name"] == selected_name].iloc[0]
    product_id = selected["product_id"]
    unit_price = int(selected["unit_price"])

    product_df = daily_df[daily_df["product_id"] == product_id]
    latest_date = get_latest_product_sales_date(daily_df, product_id) or get_latest_business_date(
        daily_df, day_totals
    )
    if latest_date is None:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    latest_row = product_df[product_df["date"] == to_ts(latest_date)]
    latest_units = int(latest_row["units_sold"].iloc[0]) if not latest_row.empty else 0
    single_revenue = latest_units * unit_price

    lw_sales = last_week_same_day_sales(daily_df, product_id, latest_date)
    avg_4w = four_week_same_weekday_avg(daily_df, product_id, latest_date)
    selection_rate = calc_food_selection_rate(daily_df, product_id, day_totals, days=7)

    latest_label = latest_date.strftime("%Y/%m/%d")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"単品売上高（{latest_label}）", f"¥{single_revenue:,}")
    with m2:
        st.metric(f"販売数（{latest_label}）", f"{latest_units:,} 個")
    with m3:
        st.metric("前週同曜日販売数", f"{lw_sales:,} 個")
    with m4:
        st.metric("フード選択率（直近7日）", f"{selection_rate:.2%}")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(plot_weekday_comparison(lw_sales, avg_4w, selected_name), use_container_width=True)
        st.caption(f"過去1か月同曜日平均: **{avg_4w:,.1f} 個**")
    with c2:
        trend_df = product_df[product_df["date"] >= to_ts(latest_date - timedelta(days=30))]
        st.plotly_chart(plot_daily_trend(trend_df, selected_name), use_container_width=True)

    st.markdown('<p class="section-title">発注予測シミュレーター</p>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    predicted_customers = fc1.number_input("予測客数", min_value=100, max_value=6000, value=2400, step=50)
    correction_pct = fc2.slider("天気・イベント補正（%）", -30, 50, 0)
    current_stock = fc3.number_input("現在の在庫数（個）", min_value=0, max_value=5000, value=120, step=5)

    correction_factor = 1.0 + correction_pct / 100.0
    recommended = max(0, int(round(predicted_customers * selection_rate * correction_factor - current_stock)))

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("補正係数", f"×{correction_factor:.2f}")
    with r2:
        st.metric("理論需要数", f"{int(predicted_customers * selection_rate * correction_factor):,} 個")
    with r3:
        st.metric("推奨発注量", f"{recommended:,} 個")

    st.code(
        f"推奨発注量 = (予測客数 {predicted_customers:,} × フード選択率 {selection_rate:.4f}) "
        f"× 補正係数 {correction_factor:.2f} − 現在在庫 {current_stock}\n"
        f"         = {recommended:,} 個",
        language="text",
    )

    if recommended > COLD_STORAGE_CAPACITY:
        st.error(f"推奨発注量が限界収容量（{COLD_STORAGE_CAPACITY:,} 個）を超えています。")
    elif recommended > COLD_STORAGE_CAPACITY * 0.85:
        st.warning("収容量の85%を超えています。")
    else:
        st.success("収容容量内の推奨発注量です。")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="wide")
    inject_css()
    ensure_data_files()

    products_df = load_products()
    daily_df = load_daily_sales()

    st.markdown(
        f"""
        <div class="main-title">
            <h1>🍽️ {APP_TITLE}</h1>
            <p>Square CSV（ルートB）+ 手入力 · ダッシュボード・発注予測 · 日商目安 ¥{DAILY_REVENUE_TARGET:,}+</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_daily_data_empty(daily_df):
        show_empty_data_notice()

    with st.sidebar:
        st.markdown("### データ管理")
        st.caption(f"登録日数: **{len(list_recorded_dates(daily_df))}** 日")
        if st.button("全データをリセット", type="secondary", use_container_width=True):
            reset_application_data()
            st.success("データを初期化しました。")
            st.rerun()

    t1, t2, t3, t4 = st.tabs(
        ["Square CSVアップロード", "日次データ入力", "過去データ確認・修正", "ダッシュボード・可視化"]
    )
    with t1:
        render_square_upload_tab(products_df, daily_df)
    with t2:
        render_daily_input_tab(products_df, daily_df)
    with t3:
        render_history_tab(products_df, daily_df)
    with t4:
        render_dashboard_tab(products_df, daily_df)


if __name__ == "__main__":
    main()
