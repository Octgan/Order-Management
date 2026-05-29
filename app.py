"""
手動入力型 フード発注管理アプリ
- データ入力タブ: 日次営業データの保存
- 商品管理タブ: 新商品追加 / 販売終了
- ダッシュボードタブ: KPI可視化 / 発注予測
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
APP_TITLE = "手動入力型 フード発注管理"
DAILY_REVENUE_TARGET = 1_000_000
COLD_STORAGE_CAPACITY = 300

DATA_DIR = Path(__file__).resolve().parent / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
DAILY_CSV = DATA_DIR / "daily_sales.csv"

DEFAULT_PRODUCTS: list[dict[str, Any]] = [
    {"name": "ワッフル", "unit_price": 900},
    {"name": "パフェ", "unit_price": 1250},
    {"name": "チーズケーキ", "unit_price": 780},
    {"name": "抹茶チーズケーキ", "unit_price": 850},
    {"name": "バナナパウンドケーキ", "unit_price": 720},
    {"name": "レモンパウンドケーキ", "unit_price": 720},
    {"name": "グラノーラ", "unit_price": 760},
    {"name": "オーバーナイトグラノーラ", "unit_price": 840},
    {"name": "季節のソースとグラノーラ", "unit_price": 980},
]

BASE_QTY_BY_NAME: dict[str, int] = {
    "ワッフル": 220,
    "パフェ": 170,
    "チーズケーキ": 185,
    "抹茶チーズケーキ": 150,
    "バナナパウンドケーキ": 120,
    "レモンパウンドケーキ": 110,
    "グラノーラ": 95,
    "オーバーナイトグラノーラ": 85,
    "季節のソースとグラノーラ": 105,
}


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
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PRODUCTS_CSV.exists():
        now = datetime.now().isoformat()
        rows = []
        for idx, p in enumerate(DEFAULT_PRODUCTS, start=1):
            rows.append(
                {
                    "product_id": f"FOOD_{idx:03d}",
                    "name": p["name"],
                    "unit_price": int(p["unit_price"]),
                    "is_active": 1,
                    "created_at": now,
                }
            )
        pd.DataFrame(rows).to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")

    if not DAILY_CSV.exists():
        pd.DataFrame(
            columns=[
                "date",
                "total_sales",
                "total_customers",
                "product_id",
                "product_name",
                "unit_price",
                "units_sold",
                "created_at",
            ]
        ).to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")

    # 常に最低1か月分のダミー日次データを補完
    seed_initial_data(days=35, seed=42)


def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(0).astype(int)
    return df


def load_daily_sales() -> pd.DataFrame:
    if DAILY_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(DAILY_CSV, encoding="utf-8-sig")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    numeric_cols = ["total_sales", "total_customers", "unit_price", "units_sold"]
    for col in numeric_cols:
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

    # 同日データは上書き
    if not daily_df.empty:
        daily_df = daily_df[daily_df["date"] != target_ts]

    rows = []
    for _, row in products_df.iterrows():
        name = row["name"]
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

    new_df = pd.DataFrame(rows)
    merged = pd.concat([daily_df, new_df], ignore_index=True)
    merged.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")


def get_daily_record_by_date(daily_df: pd.DataFrame, target_date: date) -> tuple[dict[str, int], dict[str, int]]:
    """指定日の総売上/総客数と商品別販売数を返す。"""
    if daily_df.empty:
        return {"total_sales": 0, "total_customers": 0}, {}
    target_ts = to_ts(target_date)
    day_rows = daily_df[daily_df["date"] == target_ts].copy()
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
    summary = summary.rename(
        columns={
            "date": "日付",
            "total_sales": "店舗総売上",
            "total_customers": "総客数",
            "total_food_units": "フード販売合計",
        }
    )
    return summary


def add_product(name: str, unit_price: int) -> None:
    products_df = load_products()
    existing_names = set(products_df["name"].astype(str).tolist())
    if name in existing_names:
        # 既存商品が非アクティブなら復帰
        products_df.loc[products_df["name"] == name, "is_active"] = 1
        products_df.loc[products_df["name"] == name, "unit_price"] = int(unit_price)
    else:
        ids = products_df["product_id"].astype(str).tolist()
        next_num = len(ids) + 1
        while f"FOOD_{next_num:03d}" in ids:
            next_num += 1
        new_row = pd.DataFrame(
            [
                {
                    "product_id": f"FOOD_{next_num:03d}",
                    "name": name,
                    "unit_price": int(unit_price),
                    "is_active": 1,
                    "created_at": datetime.now().isoformat(),
                }
            ]
        )
        products_df = pd.concat([products_df, new_row], ignore_index=True)

    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def deactivate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 0
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def activate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 1
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def seed_initial_data(days: int = 35, seed: int = 42) -> None:
    """初回起動時に約1か月分のリアルなダミーデータを投入する。"""
    rng = np.random.default_rng(seed)
    products_df = load_products()
    daily_df = load_daily_sales()
    existing_dates = set()
    if not daily_df.empty:
        existing_dates = set(daily_df["date"].dt.date.tolist())

    for d_offset in range(days, -1, -1):
        d = date.today() - timedelta(days=d_offset)
        if d in existing_dates:
            continue
        weekday = d.weekday()
        day_factor = 1.0 + (0.22 if weekday >= 4 else 0.0) + (0.07 if weekday == 5 else 0.10 if weekday == 6 else 0.0)
        noise = float(rng.normal(1.0, 0.07))
        total_customers = int(1900 * day_factor * noise)

        units_by_product: dict[str, int] = {}
        calc_food_sales = 0
        for _, p in products_df.iterrows():
            base = BASE_QTY_BY_NAME.get(str(p["name"]), 85)
            units = max(25, int(base * day_factor * noise * float(rng.uniform(0.90, 1.12))))
            units_by_product[str(p["name"])] = units
            calc_food_sales += units * int(p["unit_price"])

        # フード以外売上(ドリンク等)を加味し、日商100万規模に寄せる
        beverage_sales = int(total_customers * float(rng.uniform(290, 360)))
        total_sales = int(calc_food_sales + beverage_sales)
        save_daily_input(d, total_sales, total_customers, units_by_product, products_df)


def get_day_totals(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "total_sales", "total_customers"])
    day_totals = (
        daily_df.sort_values("date")
        .groupby("date", as_index=False)[["total_sales", "total_customers"]]
        .first()
    )
    return day_totals


def get_same_weekday_df(daily_df: pd.DataFrame, target_date: date, product_id: str) -> pd.DataFrame:
    ref_ts = to_ts(target_date)
    wd = ref_ts.weekday()
    mask = (
        (daily_df["product_id"] == product_id)
        & (daily_df["date"].dt.weekday == wd)
        & (daily_df["date"] < ref_ts)
    )
    return daily_df.loc[mask].copy()


def last_week_same_day_sales(daily_df: pd.DataFrame, product_id: str, ref: date) -> int:
    last_week_ts = to_ts(ref - timedelta(days=7))
    row = daily_df[(daily_df["date"] == last_week_ts) & (daily_df["product_id"] == product_id)]
    return int(row["units_sold"].iloc[0]) if not row.empty else 0


def four_week_same_weekday_avg(daily_df: pd.DataFrame, product_id: str, ref: date) -> float:
    hist = get_same_weekday_df(daily_df, ref, product_id)
    cutoff = to_ts(ref - timedelta(days=28))
    last_4 = hist[hist["date"] >= cutoff]
    if last_4.empty:
        return 0.0
    return float(last_4["units_sold"].mean())


def calc_food_selection_rate(daily_df: pd.DataFrame, product_id: str, day_totals: pd.DataFrame, days: int = 7) -> float:
    if daily_df.empty or day_totals.empty:
        return 0.0
    end = to_ts(date.today())
    start = to_ts(date.today() - timedelta(days=days))
    sub = daily_df[
        (daily_df["product_id"] == product_id)
        & (daily_df["date"] >= start)
        & (daily_df["date"] <= end)
    ]
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
        showlegend=False,
        height=360,
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
        hovermode="x unified",
        height=400,
    )
    return fig


def render_data_input_tab(products_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">日々の営業データ入力</p>', unsafe_allow_html=True)
    active_df = products_df[products_df["is_active"] == 1].copy()
    if active_df.empty:
        st.warning("有効な商品がありません。先に「商品管理」タブで商品を追加してください。")
        return

    daily_df = load_daily_sales()
    with st.form("daily_input_form"):
        target_date = st.date_input("日付", value=date.today())
        c1, c2 = st.columns(2)
        with c1:
            total_sales = st.number_input("店舗全体の売上高（円）", min_value=0, value=1_000_000, step=10_000)
        with c2:
            total_customers = st.number_input("総客数（人）", min_value=0, value=2_000, step=10)

        st.markdown("#### 商品別の販売個数")
        units_by_product: dict[str, int] = {}
        col_left, col_right = st.columns(2)
        active_rows = list(active_df.itertuples(index=False))
        half = (len(active_rows) + 1) // 2
        left_rows = active_rows[:half]
        right_rows = active_rows[half:]

        with col_left:
            for p in left_rows:
                units_by_product[p.name] = int(
                    st.number_input(
                        f"{p.name}（個）",
                        min_value=0,
                        value=max(20, BASE_QTY_BY_NAME.get(str(p.name), 80)),
                        step=1,
                        key=f"units_left_{p.product_id}",
                    )
                )
        with col_right:
            for p in right_rows:
                units_by_product[p.name] = int(
                    st.number_input(
                        f"{p.name}（個）",
                        min_value=0,
                        value=max(20, BASE_QTY_BY_NAME.get(str(p.name), 80)),
                        step=1,
                        key=f"units_right_{p.product_id}",
                    )
                )

        submitted = st.form_submit_button("データを保存する", type="primary", use_container_width=True)

    if submitted:
        if not daily_df.empty and (daily_df["date"] == to_ts(target_date)).any():
            st.warning(f"{target_date} は既存データがあるため、上書き保存しました。")
        save_daily_input(target_date, int(total_sales), int(total_customers), units_by_product, active_df)
        st.success(f"{target_date} の営業データを保存しました。")
        st.rerun()


def render_history_edit_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">過去データの確認・修正</p>', unsafe_allow_html=True)
    if daily_df.empty:
        st.info("まだ営業データがありません。先に「データ入力」タブから登録してください。")
        return

    summary_table = make_daily_summary_table(daily_df)
    st.markdown("#### 登録済みデータ一覧")
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    available_dates = sorted(daily_df["date"].dt.date.unique().tolist(), reverse=True)
    selected_date = st.selectbox("修正する日付", available_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
    totals, units_map = get_daily_record_by_date(daily_df, selected_date)

    st.caption("選択日のデータを編集して「データを更新する」で上書き保存します。")
    with st.form("history_edit_form"):
        c1, c2 = st.columns(2)
        with c1:
            edit_total_sales = st.number_input(
                "店舗総売上（円）",
                min_value=0,
                value=int(totals.get("total_sales", 0)),
                step=10_000,
            )
        with c2:
            edit_total_customers = st.number_input(
                "総客数（人）",
                min_value=0,
                value=int(totals.get("total_customers", 0)),
                step=10,
            )

        st.markdown("#### 商品別の販売個数（修正）")
        edit_units_by_product: dict[str, int] = {}
        left_col, right_col = st.columns(2)
        product_rows = list(products_df.sort_values("product_id").itertuples(index=False))
        half = (len(product_rows) + 1) // 2
        left_rows = product_rows[:half]
        right_rows = product_rows[half:]

        with left_col:
            for p in left_rows:
                default_val = int(units_map.get(str(p.name), 0))
                edit_units_by_product[p.name] = int(
                    st.number_input(
                        f"{p.name}（個）",
                        min_value=0,
                        value=default_val,
                        step=1,
                        key=f"edit_units_left_{selected_date}_{p.product_id}",
                    )
                )
        with right_col:
            for p in right_rows:
                default_val = int(units_map.get(str(p.name), 0))
                edit_units_by_product[p.name] = int(
                    st.number_input(
                        f"{p.name}（個）",
                        min_value=0,
                        value=default_val,
                        step=1,
                        key=f"edit_units_right_{selected_date}_{p.product_id}",
                    )
                )

        updated = st.form_submit_button("データを更新する（上書き保存）", type="primary", use_container_width=True)

    if updated:
        save_daily_input(
            selected_date,
            int(edit_total_sales),
            int(edit_total_customers),
            edit_units_by_product,
            products_df,
        )
        st.success(f"{selected_date} のデータを更新しました。ダッシュボードにも即反映されます。")
        st.rerun()


def render_product_tab(products_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">商品管理</p>', unsafe_allow_html=True)
    st.caption("新メニュー追加、販売終了（非表示）、再開ができます。")

    with st.form("add_product_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            new_name = st.text_input("新商品名")
        with c2:
            new_price = st.number_input("単価（円）", min_value=0, value=800, step=10)
        add_submit = st.form_submit_button("商品を追加・登録", type="primary")

    if add_submit:
        if not new_name.strip():
            st.error("商品名を入力してください。")
        else:
            add_product(new_name.strip(), int(new_price))
            st.success(f"商品を登録しました: {new_name.strip()}")
            st.rerun()

    st.markdown("#### 現在の商品一覧")
    view_df = products_df.copy()
    view_df["状態"] = np.where(view_df["is_active"] == 1, "販売中", "販売終了")
    st.dataframe(view_df[["product_id", "name", "unit_price", "状態"]], use_container_width=True, hide_index=True)

    active_df = products_df[products_df["is_active"] == 1]
    inactive_df = products_df[products_df["is_active"] == 0]

    c1, c2 = st.columns(2)
    with c1:
        if not active_df.empty:
            selected_off = st.selectbox(
                "販売終了にする商品",
                active_df["product_id"] + " | " + active_df["name"],
                key="deactivate_select",
            )
            if st.button("販売終了にする", use_container_width=True):
                pid = selected_off.split(" | ")[0]
                deactivate_product(pid)
                st.success("商品を販売終了にしました。")
                st.rerun()

    with c2:
        if not inactive_df.empty:
            selected_on = st.selectbox(
                "販売再開する商品",
                inactive_df["product_id"] + " | " + inactive_df["name"],
                key="activate_select",
            )
            if st.button("販売再開する", use_container_width=True):
                pid = selected_on.split(" | ")[0]
                activate_product(pid)
                st.success("商品を販売再開しました。")
                st.rerun()


def render_dashboard_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">ダッシュボード・可視化</p>', unsafe_allow_html=True)
    if daily_df.empty:
        st.warning("営業データがありません。先に「データ入力」タブで保存してください。")
        return

    daily_df = daily_df.sort_values("date")
    day_totals = get_day_totals(daily_df)
    latest_date = day_totals["date"].max().date()

    product_options = products_df["name"].tolist()
    selected_name = st.selectbox("対象フード商品", product_options, key="dashboard_product")
    selected = products_df[products_df["name"] == selected_name].iloc[0]
    product_id = selected["product_id"]
    unit_price = int(selected["unit_price"])

    product_df = daily_df[daily_df["product_id"] == product_id].copy()
    latest_row = product_df[product_df["date"] == to_ts(latest_date)]
    latest_units = int(latest_row["units_sold"].iloc[0]) if not latest_row.empty else 0
    single_revenue = latest_units * unit_price

    lw_sales = last_week_same_day_sales(daily_df, product_id, latest_date)
    avg_4w = four_week_same_weekday_avg(daily_df, product_id, latest_date)
    selection_rate = calc_food_selection_rate(daily_df, product_id, day_totals, days=7)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("単品売上高（最新日）", f"¥{single_revenue:,}")
    with m2:
        st.metric("最新日の販売数", f"{latest_units:,} 個")
    with m3:
        st.metric("前週同曜日販売数", f"{lw_sales:,} 個")
    with m4:
        st.metric("フード選択率", f"{selection_rate:.2%}")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.plotly_chart(plot_weekday_comparison(lw_sales, avg_4w, selected_name), use_container_width=True)
        st.caption(f"過去1か月同曜日平均: **{avg_4w:,.1f} 個**")
    with c2:
        trend_df = product_df[product_df["date"] >= to_ts(latest_date - timedelta(days=30))]
        st.plotly_chart(plot_daily_trend(trend_df, selected_name), use_container_width=True)

    st.markdown('<p class="section-title">発注予測シミュレーター</p>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        predicted_customers = st.number_input("予測客数（または目標客数）", min_value=100, max_value=6000, value=2400, step=50)
    with fc2:
        correction_pct = st.slider("天気・イベント補正（%）", min_value=-30, max_value=50, value=0)
    with fc3:
        current_stock = st.number_input("現在の在庫数（個）", min_value=0, max_value=5000, value=120, step=5)

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
        st.error(
            f"⚠️ 推奨発注量（{recommended:,} 個）が限界収容量（{COLD_STORAGE_CAPACITY:,} 個）を "
            f"{recommended - COLD_STORAGE_CAPACITY:,} 個超えています。"
        )
    elif recommended > COLD_STORAGE_CAPACITY * 0.85:
        st.warning("収容量の85%を超えています。保管スペースと欠品リスクを再確認してください。")
    else:
        st.success("収容量内の推奨発注量です。")


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
            <p>手動入力データ蓄積型（CSV保存） · 日商目安 ¥{DAILY_REVENUE_TARGET:,}+</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["データ入力", "過去データ確認・修正", "商品管理", "ダッシュボード・可視化"])
    with t1:
        render_data_input_tab(products_df)
    with t2:
        render_history_edit_tab(products_df, daily_df)
    with t3:
        render_product_tab(products_df)
    with t4:
        render_dashboard_tab(products_df, daily_df)


if __name__ == "__main__":
    main()
