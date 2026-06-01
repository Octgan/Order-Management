"""
フード発注管理アプリ
- Square CSVルートB（2枚同時アップロード）で日次データ取り込み
- 過去データの確認・修正
- ダッシュボード・発注予測
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from square_csv import (
    DayImport,
    build_square_product_label,
    is_meaningful_day,
    list_square_labels_from_matrix,
    load_uploaded_dataframe,
    parse_dual_csv_upload,
    read_uploaded_csv,
    set_custom_product_mappings,
    summarize_square_row_mapping,
)

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
APP_TITLE = "Square CSV連携 フード発注管理"
COLD_STORAGE_CAPACITY = 300
DATA_VERSION = 5  # 仕様変更時に上げると data/ を初期化（4→5は税抜へ戻す）
STANDARD_CONSUMPTION_TAX_RATE = 0.10

DATA_DIR = Path(__file__).resolve().parent / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
MAPPINGS_CSV = DATA_DIR / "product_mappings.csv"
DAILY_CSV = DATA_DIR / "daily_sales.csv"
VERSION_FILE = DATA_DIR / ".data_version"

MAPPING_COLUMNS = ["square_label", "product_name", "created_at"]

DEFAULT_PRODUCTS: list[dict[str, Any]] = [
    {"name": "ワッフル", "unit_price": 900},
    {"name": "パフェ", "unit_price": 1250},
    {"name": "抹茶ケーキ", "unit_price": 850},
    {"name": "チーズケーキ", "unit_price": 780},
    {"name": "バナナパウンドケーキ", "unit_price": 720},
    {"name": "レモンケーキ", "unit_price": 720},
    {"name": "ミートパイ", "unit_price": 650},
    {"name": "レモンパイ", "unit_price": 650},
]

EMPTY_DATA_MESSAGE = (
    "データがありません。「Square CSVアップロード」タブからCSVを取り込んでください。"
)

WEEKDAY_LABELS_JA = ["月", "火", "水", "木", "金", "土", "日"]

DAILY_SALES_COLUMNS = [
    "date",
    "total_sales",
    "total_customers",
    "product_id",
    "product_name",
    "unit_price",
    "units_sold",
    "product_sales",
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


def tax_exclusive_price(inclusive_yen: int, rate: float = STANDARD_CONSUMPTION_TAX_RATE) -> int:
    """税込金額を税抜（四捨五入）に換算。"""
    if inclusive_yen <= 0:
        return 0
    return int(round(inclusive_yen / (1 + rate)))


def migrate_v4_to_tax_exclusive() -> None:
    """v4→v5: 税込にしていた単価・保存済み売上を税抜に戻す。以降CSVは純売上高を使用。"""
    products_df = load_products()
    products_df["unit_price"] = products_df["unit_price"].apply(
        lambda p: tax_exclusive_price(int(p))
    )
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")

    daily_df = load_daily_sales()
    if not daily_df.empty:
        price_by_id = dict(zip(products_df["product_id"], products_df["unit_price"]))
        daily_df["unit_price"] = daily_df["product_id"].map(price_by_id).fillna(daily_df["unit_price"])
        for day_val, group in daily_df.groupby(daily_df["date"].dt.date):
            mask = daily_df["date"].dt.date == day_val
            old_total = int(group["total_sales"].iloc[0])
            new_total = tax_exclusive_price(old_total)
            daily_df.loc[mask, "total_sales"] = new_total
            if "product_sales" in daily_df.columns:
                daily_df.loc[mask, "product_sales"] = (
                    pd.to_numeric(daily_df.loc[mask, "product_sales"], errors="coerce")
                    .fillna(0)
                    .apply(tax_exclusive_price)
                    .astype(int)
                )
        daily_df.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")

    VERSION_FILE.write_text(str(DATA_VERSION), encoding="utf-8")


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
    """営業データと商品マスタを初期状態に戻す（紐づけ設定は保持）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_default_products()
    init_empty_daily_csv()
    if not MAPPINGS_CSV.exists():
        init_empty_mappings_csv()
    VERSION_FILE.write_text(str(DATA_VERSION), encoding="utf-8")
    sync_square_mappings()


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stored_version = 0
    if VERSION_FILE.exists():
        try:
            stored_version = int(VERSION_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            stored_version = 0

    if stored_version < DATA_VERSION:
        if stored_version == 4 and DATA_VERSION >= 5:
            migrate_v4_to_tax_exclusive()
        elif stored_version == 3 and DATA_VERSION >= 5:
            VERSION_FILE.write_text(str(DATA_VERSION), encoding="utf-8")
        else:
            reset_application_data()
        return

    if not PRODUCTS_CSV.exists():
        write_default_products()
    if not DAILY_CSV.exists():
        init_empty_daily_csv()
    if not MAPPINGS_CSV.exists():
        init_empty_mappings_csv()
    sync_square_mappings()


def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int)
    return df


def active_products(products_df: pd.DataFrame) -> pd.DataFrame:
    active = products_df[products_df["is_active"] == 1].copy()
    return active if not active.empty else products_df.copy()


def init_empty_mappings_csv() -> None:
    pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(MAPPINGS_CSV, index=False, encoding="utf-8-sig")


def load_product_mappings_df() -> pd.DataFrame:
    if not MAPPINGS_CSV.exists() or MAPPINGS_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    return pd.read_csv(MAPPINGS_CSV, encoding="utf-8-sig")


def load_product_mappings() -> dict[str, str]:
    """square_label → product_name"""
    df = load_product_mappings_df()
    if df.empty:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        label = str(row["square_label"]).strip()
        name = str(row["product_name"]).strip()
        if label and name:
            out[label] = name
    return out


def sync_square_mappings() -> None:
    set_custom_product_mappings(load_product_mappings())


def save_product_mapping(square_label: str, product_name: str) -> None:
    label = square_label.strip()
    name = product_name.strip()
    if not label or not name:
        raise ValueError("Square表記とアプリ商品名を入力してください。")
    df = load_product_mappings_df()
    df = df[df["square_label"].astype(str).str.strip() != label]
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [{"square_label": label, "product_name": name, "created_at": datetime.now().isoformat()}]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(MAPPINGS_CSV, index=False, encoding="utf-8-sig")
    sync_square_mappings()


def delete_product_mapping(square_label: str) -> None:
    label = square_label.strip()
    df = load_product_mappings_df()
    df = df[df["square_label"].astype(str).str.strip() != label]
    if df.empty:
        init_empty_mappings_csv()
    else:
        df.to_csv(MAPPINGS_CSV, index=False, encoding="utf-8-sig")
    sync_square_mappings()


def add_product(name: str, unit_price: int) -> None:
    products_df = load_products()
    name = name.strip()
    if not name:
        raise ValueError("商品名を入力してください。")
    if name in set(products_df["name"].astype(str)):
        products_df.loc[products_df["name"] == name, "is_active"] = 1
        products_df.loc[products_df["name"] == name, "unit_price"] = int(unit_price)
    else:
        ids = products_df["product_id"].astype(str).tolist()
        next_num = len(ids) + 1
        while f"FOOD_{next_num:03d}" in ids:
            next_num += 1
        products_df = pd.concat(
            [
                products_df,
                pd.DataFrame(
                    [
                        {
                            "product_id": f"FOOD_{next_num:03d}",
                            "name": name,
                            "unit_price": int(unit_price),
                            "is_active": 1,
                            "created_at": datetime.now().isoformat(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def update_product_price(product_id: str, unit_price: int) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "unit_price"] = int(unit_price)
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def deactivate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 0
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def activate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 1
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def load_daily_sales() -> pd.DataFrame:
    if not DAILY_CSV.exists() or DAILY_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(DAILY_CSV, encoding="utf-8-sig")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    for col in ["total_sales", "total_customers", "unit_price", "units_sold", "product_sales"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def allocate_product_sales_from_store(
    total_sales: int, units_by_product: dict[str, int]
) -> dict[str, int]:
    """日別保存時: 店舗総売上を登録フードの販売数比率で按分。"""
    total_units = sum(units_by_product.values())
    if total_sales <= 0 or total_units <= 0:
        return {name: 0 for name in units_by_product}
    return {
        name: int(total_sales * units / total_units)
        for name, units in units_by_product.items()
    }


def calc_single_item_sales_indicator(store_total_sales: int, units_sold: int) -> int:
    """単品売上高（税抜）= その日の店舗純売上 ÷ 当該商品の販売個数。"""
    if units_sold <= 0 or store_total_sales <= 0:
        return 0
    return int(store_total_sales / units_sold)


def get_product_line_sales(
    daily_df: pd.DataFrame,
    target_date: date,
    product_id: str,
    product_name: str,
) -> int:
    """その日の当該商品売上（¥）。保存値がなければ店舗売上を販売数比率で按分。"""
    target_ts = to_ts(target_date)
    day_rows = daily_df[daily_df["date"] == target_ts]
    if day_rows.empty:
        return 0
    row = day_rows[day_rows["product_id"] == product_id]
    if row.empty:
        return 0
    stored = int(row["product_sales"].iloc[0]) if "product_sales" in row.columns else 0
    if stored > 0:
        return stored
    total_sales = int(day_rows["total_sales"].iloc[0])
    units = int(row["units_sold"].iloc[0])
    all_units = int(day_rows["units_sold"].sum())
    if total_sales > 0 and all_units > 0:
        return int(total_sales * units / all_units)
    return 0


def save_daily_input(
    target_date: date,
    total_sales: int,
    total_customers: int,
    units_by_product: dict[str, int],
    products_df: pd.DataFrame,
    product_sales_by_product: dict[str, int] | None = None,
) -> None:
    daily_df = load_daily_sales()
    target_ts = to_ts(target_date)
    now = datetime.now().isoformat()

    if not daily_df.empty:
        daily_df = daily_df[daily_df["date"] != target_ts]

    if product_sales_by_product is None:
        product_sales_by_product = allocate_product_sales_from_store(total_sales, units_by_product)

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
                "product_sales": int(product_sales_by_product.get(name, 0)),
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


def list_recorded_dates(daily_df: pd.DataFrame) -> list[date]:
    if daily_df.empty:
        return []
    return sorted(daily_df["date"].dt.date.unique(), reverse=True)


def purge_meaningless_days() -> int:
    """販売0・売上/客数が極小の日を daily_sales から削除。戻り値=削除した日数。"""
    daily_df = load_daily_sales()
    if daily_df.empty:
        return 0
    removed = 0
    keep_parts: list[pd.DataFrame] = []
    for day_val, group in daily_df.groupby(daily_df["date"].dt.date):
        units = dict(zip(group["product_name"].astype(str), group["units_sold"].astype(int)))
        summary = None
        if not group.empty:
            from square_csv import DaySummary

            summary = DaySummary(
                int(group["total_sales"].iloc[0]),
                int(group["total_customers"].iloc[0]),
            )
        if is_meaningful_day(units, summary):
            keep_parts.append(group)
        else:
            removed += 1
    if removed == 0:
        return 0
    if keep_parts:
        pd.concat(keep_parts, ignore_index=True).to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")
    else:
        init_empty_daily_csv()
    return removed


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
        sales = {}
        if day_data.product_sales_by_product:
            sales = {
                name: day_data.product_sales_by_product.get(name, 0)
                for name in product_list["name"].astype(str)
            }
        save_daily_input(
            day_data.day,
            day_data.total_sales,
            day_data.total_customers,
            units,
            product_list,
            product_sales_by_product=sales or None,
        )

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


def calc_avg_units_sold_in_period(
    daily_df: pd.DataFrame,
    product_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, float | int]:
    """指定期間の販売個数統計（データ登録がある日のみ）。"""
    if start_date > end_date or daily_df.empty:
        return {"avg": 0.0, "total": 0, "days": 0, "active_days": 0}
    sub = daily_df[
        (daily_df["product_id"] == product_id)
        & (daily_df["date"] >= to_ts(start_date))
        & (daily_df["date"] <= to_ts(end_date))
    ]
    if sub.empty:
        return {"avg": 0.0, "total": 0, "days": 0, "active_days": 0}
    units = sub["units_sold"].astype(int)
    days = int(len(sub))
    total = int(units.sum())
    active_days = int((units > 0).sum())
    return {
        "avg": float(units.mean()),
        "total": total,
        "days": days,
        "active_days": active_days,
    }


def calc_avg_units_by_weekday_in_period(
    daily_df: pd.DataFrame,
    product_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """指定期間の曜日別平均販売個数（登録がある日のみ、月〜日）。"""
    columns = ["weekday", "曜日", "avg_units", "days", "total_units"]
    if start_date > end_date or daily_df.empty:
        return pd.DataFrame(columns=columns)
    sub = daily_df[
        (daily_df["product_id"] == product_id)
        & (daily_df["date"] >= to_ts(start_date))
        & (daily_df["date"] <= to_ts(end_date))
    ][["date", "units_sold"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=columns)
    sub["weekday"] = sub["date"].dt.weekday
    grouped = (
        sub.groupby("weekday", as_index=False)
        .agg(avg_units=("units_sold", "mean"), days=("units_sold", "count"), total_units=("units_sold", "sum"))
        .sort_values("weekday")
    )
    grouped["曜日"] = grouped["weekday"].map(lambda w: WEEKDAY_LABELS_JA[int(w)])
    return grouped[columns]


def plot_weekday_avg_units(weekday_df: pd.DataFrame, product_name: str, period_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekday_df["曜日"],
            y=weekday_df["avg_units"],
            marker_color="#0f3460",
            text=[f"{v:,.1f} 個" for v in weekday_df["avg_units"]],
            textposition="outside",
            customdata=np.stack([weekday_df["days"], weekday_df["total_units"]], axis=-1),
            hovertemplate=(
                "曜日: %{x}<br>"
                "平均販売数: %{y:,.1f} 個<br>"
                "集計日数: %{customdata[0]} 日<br>"
                "合計: %{customdata[1]:,} 個<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{product_name} — 曜日別の平均販売個数（{period_label}）",
        xaxis_title="曜日",
        yaxis_title="平均販売数（個/日）",
        template="plotly_white",
        height=360,
        showlegend=False,
    )
    return fig


def calc_avg_customers_in_period(
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[str, float | int]:
    """指定期間の店舗客数統計（登録がある日のみ）。"""
    if start_date > end_date or day_totals.empty:
        return {"avg": 0.0, "total": 0, "days": 0, "active_days": 0}
    sub = day_totals[
        (day_totals["date"] >= to_ts(start_date)) & (day_totals["date"] <= to_ts(end_date))
    ]
    if sub.empty:
        return {"avg": 0.0, "total": 0, "days": 0, "active_days": 0}
    customers = sub["total_customers"].astype(int)
    days = int(len(sub))
    total = int(customers.sum())
    active_days = int((customers > 0).sum())
    return {
        "avg": float(customers.mean()),
        "total": total,
        "days": days,
        "active_days": active_days,
    }


def build_customer_trend_frame(
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if start_date > end_date or day_totals.empty:
        return pd.DataFrame(columns=["date", "total_customers"])
    sub = day_totals[
        (day_totals["date"] >= to_ts(start_date)) & (day_totals["date"] <= to_ts(end_date))
    ][["date", "total_customers"]].copy()
    return sub.sort_values("date")


def plot_customer_trend(customer_df: pd.DataFrame, period_avg: float, period_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=customer_df["date"],
            y=customer_df["total_customers"],
            mode="lines+markers",
            name="日次客数",
            line=dict(color="#16a085", width=3),
            marker=dict(size=7),
            hovertemplate="日付: %{x|%Y-%m-%d}<br>客数: %{y:,}人<extra></extra>",
        )
    )
    if period_avg > 0:
        fig.add_hline(
            y=period_avg,
            line_dash="dash",
            line_color="#e94560",
            annotation_text=f"期間平均 {period_avg:,.1f} 人",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"店舗客数の推移（{period_label}）",
        xaxis_title="日付",
        yaxis_title="客数（人）",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def resolve_dashboard_period(
    period_preset: str,
    *,
    data_min: date,
    data_max: date,
    latest_date: date,
) -> tuple[date, date]:
    if period_preset == "直近7日":
        return max(data_min, latest_date - timedelta(days=6)), latest_date
    if period_preset == "直近30日":
        return max(data_min, latest_date - timedelta(days=29)), latest_date
    if period_preset == "直近90日":
        return max(data_min, latest_date - timedelta(days=89)), latest_date
    return data_min, data_max


def calc_food_selection_rate(
    daily_df: pd.DataFrame,
    product_id: str,
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> float:
    """指定期間のフード選択率 = 対象商品の販売数合計 ÷ 店舗客数合計。"""
    if daily_df.empty or day_totals.empty or start_date > end_date:
        return 0.0
    start = to_ts(start_date)
    end = to_ts(end_date)
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
        height=360,
        showlegend=False,
    )
    return fig


def build_product_trend_frame(
    daily_df: pd.DataFrame,
    product_id: str,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """日次の販売数・店舗純売上・単品売上高（店舗純売上÷販売数）をまとめる。"""
    if start_date > end_date:
        return pd.DataFrame(columns=["date", "units_sold", "total_sales", "single_item_sales"])
    product_part = daily_df[
        (daily_df["product_id"] == product_id)
        & (daily_df["date"] >= to_ts(start_date))
        & (daily_df["date"] <= to_ts(end_date))
    ].copy()
    if product_part.empty:
        return pd.DataFrame(columns=["date", "units_sold", "total_sales", "single_item_sales"])

    store_by_day = get_day_totals(daily_df)[["date", "total_sales", "total_customers"]].rename(
        columns={"total_sales": "store_total_sales", "total_customers": "store_total_customers"}
    )
    merged = product_part[["date", "units_sold"]].merge(store_by_day, on="date", how="left")
    merged["store_total_sales"] = pd.to_numeric(merged["store_total_sales"], errors="coerce").fillna(0).astype(int)
    merged["single_item_sales"] = merged.apply(
        lambda r: calc_single_item_sales_indicator(int(r["store_total_sales"]), int(r["units_sold"])),
        axis=1,
    )
    merged["total_sales"] = merged["store_total_sales"]
    return merged.sort_values("date")


def calc_single_item_sales_stats_in_period(
    daily_df: pd.DataFrame,
    product_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, float | int]:
    """指定期間の単品売上高統計（販売数>0の日のみで平均）。"""
    trend = build_product_trend_frame(daily_df, product_id, start_date=start_date, end_date=end_date)
    if trend.empty:
        return {"avg": 0.0, "days": 0, "active_days": 0}
    active = trend[trend["units_sold"] > 0]
    days = int(len(trend))
    if active.empty:
        return {"avg": 0.0, "days": days, "active_days": 0}
    return {
        "avg": float(active["single_item_sales"].mean()),
        "days": days,
        "active_days": int(len(active)),
    }


def plot_daily_trend(product_df: pd.DataFrame, product_name: str, period_label: str = "") -> go.Figure:
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
    title_suffix = f"（{period_label}）" if period_label else ""
    fig.update_layout(
        title=f"{product_name} — 日次販売数推移{title_suffix}",
        xaxis_title="日付",
        yaxis_title="販売数（個）",
        template="plotly_white",
        height=360,
        hovermode="x unified",
    )
    return fig


def plot_single_item_sales_trend(
    trend_df: pd.DataFrame, product_name: str, period_label: str = "", period_avg: float = 0.0
) -> go.Figure:
    """単品売上高（店舗純売上÷販売数）の日次推移。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["single_item_sales"],
            mode="lines+markers",
            name="単品売上高",
            line=dict(color="#0f3460", width=3),
            customdata=np.stack(
                [trend_df["total_sales"], trend_df["units_sold"]],
                axis=-1,
            ),
            hovertemplate=(
                "日付: %{x|%Y-%m-%d}<br>"
                "単品売上高: ¥%{y:,}<br>"
                "店舗純売上: ¥%{customdata[0]:,}<br>"
                "販売数: %{customdata[1]:,}個<extra></extra>"
            ),
        )
    )
    if period_avg > 0:
        fig.add_hline(
            y=period_avg,
            line_dash="dash",
            line_color="#e94560",
            annotation_text=f"期間平均 ¥{period_avg:,.0f}",
            annotation_position="top right",
        )
    title_suffix = f"（{period_label}）" if period_label else ""
    fig.update_layout(
        title=f"{product_name} — 単品売上高の推移・税抜{title_suffix}",
        xaxis_title="日付",
        yaxis_title="単品売上高・税抜（円）",
        template="plotly_white",
        height=360,
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
        " ①商品別マトリックス、または期間の「商品売上サマリー」"
        " ②売上サマリー（客数・店舗純売上＝税抜で取り込み）"
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
                f"1. **商品別**の横持ちCSV（登録フード品目の日別販売数・売上）  \n"
                "2. **売上サマリー**のCSV（日別の総客数・店舗総売上）  \n"
                "3. 2枚をまとめてドラッグ＆ドロップ → 「データを一括取り込み」→ 下のプレビューで確認"
            )
        else:
            st.info("CSVをアップロードすると、取り込みプレビューが表示されます。")
        return

    files = uploaded_list if isinstance(uploaded_list, list) else [uploaded_list]
    if len(files) > 2:
        st.error("アップロードは2ファイルまでにしてください。")
        return

    try:
        imports, warnings = parse_dual_csv_upload(files, products_df)
    except Exception as exc:
        st.error(f"CSVの読み込みに失敗しました: {exc}")
        return

    if not imports:
        st.warning("取り込み可能な日次データがありませんでした。")
        return

    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    overlap = [d.day for d in imports if d.day in existing_dates]
    if overlap:
        st.warning(f"既存データと重複する日付: {len(overlap)}日（上書き: {'ON' if overwrite else 'OFF'}）")

    import_ready = st.button("データを一括取り込み", type="primary", use_container_width=True)
    if import_ready:
        created, updated = bulk_import_days(imports, products_df, overwrite=overwrite)
        st.success(f"取り込み完了: 新規 {created} 日 / 上書き {updated} 日")
        st.rerun()

    st.markdown("#### アップロードされたファイル")
    for f in files:
        kind, _df = load_uploaded_dataframe(f, products_df)
        label = {
            "product_matrix": "商品別マトリックス",
            "product_period_summary": "商品売上サマリー（期間）",
            "sales_summary": "売上サマリー",
            "unknown": "未判定",
        }.get(kind, kind)
        st.write(f"- **{f.name}** → {label}")

    preview_rows = [
        {
            "日付": d.day.strftime("%Y-%m-%d"),
            "店舗純売上（税抜）": d.total_sales,
            "総客数": d.total_customers,
            "フード販売合計": sum(d.units_by_product.values()),
        }
        for d in imports
    ]
    st.markdown("#### 取り込みプレビュー（日別）")
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    product_file = next(
        (
            f
            for f in files
            if load_uploaded_dataframe(f, products_df)[0]
            in ("product_matrix", "product_period_summary")
        ),
        None,
    )
    if product_file is not None:
        _kind, product_df_raw = load_uploaded_dataframe(product_file, products_df)
        mapping_df = summarize_square_row_mapping(product_df_raw, products_df)
        if not mapping_df.empty:
            st.markdown("#### 商品名の紐づけ確認（Square → アプリ）")
            st.caption(
                "「パイ」などはバリエーション列（ミート／レモンクリーム）と組み合わせて判定します。"
                " 未紐づけがある場合は取り込み前にご確認ください。"
            )
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        last_import = imports[-1]
        st.markdown(f"#### 最終日（{last_import.day}）の8品目プレビュー")
        unit_preview = pd.DataFrame(
            [{"商品": k, "販売数": v} for k, v in last_import.units_by_product.items()]
        )
        st.dataframe(unit_preview, use_container_width=True, hide_index=True)

    for msg in warnings[:8]:
        st.caption(f"⚠ {msg}")


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
                "店舗純売上・税抜（円）",
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


def render_history_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">過去データの確認・修正</p>', unsafe_allow_html=True)

    dates = list_recorded_dates(daily_df)
    if not dates:
        st.info("修正できる過去データがありません。先に Square CSV を取り込んでください。")
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


def render_products_mappings_tab(products_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">フード商品の登録</p>', unsafe_allow_html=True)
    st.caption(
        "新メニューを追加すると、CSV取り込み・ダッシュボードの対象に含められます。"
        " **単価は税抜** で登録してください（Squareの純売上・商品単価と揃えます）。"
    )

    with st.form("add_product_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            new_name = st.text_input("新しい商品名")
        with c2:
            new_price = st.number_input("単価・税抜（円）", min_value=0, value=800, step=10)
        if st.form_submit_button("商品を追加", type="primary", use_container_width=True):
            try:
                add_product(new_name, int(new_price))
                st.success(f"商品を登録しました: {new_name.strip()}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    view_df = products_df.copy()
    view_df["状態"] = np.where(view_df["is_active"] == 1, "販売中", "停止中")
    st.dataframe(
        view_df[["product_id", "name", "unit_price", "状態"]],
        use_container_width=True,
        hide_index=True,
    )

    active_df = products_df[products_df["is_active"] == 1]
    inactive_df = products_df[products_df["is_active"] == 0]
    pc1, pc2 = st.columns(2)
    with pc1:
        if not active_df.empty:
            pid = st.selectbox(
                "単価を変更する商品",
                active_df["product_id"],
                format_func=lambda x: active_df.loc[active_df["product_id"] == x, "name"].iloc[0],
                key="edit_price_product",
            )
            new_unit = st.number_input(
                "新しい単価・税抜（円）",
                min_value=0,
                value=int(active_df.loc[active_df["product_id"] == pid, "unit_price"].iloc[0]),
                step=10,
                key="edit_price_value",
            )
            if st.button("単価を更新", key="btn_update_price"):
                update_product_price(pid, int(new_unit))
                st.success("単価を更新しました。")
                st.rerun()
    with pc2:
        if not active_df.empty:
            off_id = st.selectbox(
                "販売停止",
                active_df["product_id"],
                format_func=lambda x: active_df.loc[active_df["product_id"] == x, "name"].iloc[0],
                key="deactivate_product",
            )
            if st.button("販売停止にする", key="btn_deactivate"):
                deactivate_product(off_id)
                st.rerun()
        if not inactive_df.empty:
            on_id = st.selectbox(
                "販売再開",
                inactive_df["product_id"],
                format_func=lambda x: inactive_df.loc[inactive_df["product_id"] == x, "name"].iloc[0],
                key="activate_product",
            )
            if st.button("販売再開する", key="btn_activate"):
                activate_product(on_id)
                st.rerun()

    st.markdown('<p class="section-title">Square CSV ↔ アプリ商品の紐づけ</p>', unsafe_allow_html=True)
    st.caption(
        "Squareの「商品名＋バリエーション」と、アプリの登録商品を対応づけます。"
        " ここで設定した内容がCSV取り込み時に最優先されます。"
    )

    mappings_df = load_product_mappings_df()
    if mappings_df.empty:
        st.info("まだ手動紐づけはありません。下のフォームから追加するか、CSVを読み込んで未登録行から設定してください。")
    else:
        show_map = mappings_df.rename(
            columns={"square_label": "Square表記（照合キー）", "product_name": "アプリ商品"}
        )
        st.dataframe(show_map[["Square表記（照合キー）", "アプリ商品"]], use_container_width=True, hide_index=True)
        del_label = st.selectbox(
            "削除する紐づけ",
            mappings_df["square_label"].astype(str).tolist(),
            key="delete_mapping_select",
        )
        if st.button("選択した紐づけを削除", type="secondary"):
            delete_product_mapping(del_label)
            st.success("紐づけを削除しました。")
            st.rerun()

    st.markdown("#### 紐づけを追加")
    product_names = active_products(products_df)["name"].astype(str).tolist()
    if not product_names:
        st.warning("先にフード商品を登録してください。")
        return

    tab_key, tab_parts = st.tabs(["照合キーで登録", "商品名＋バリエーションで登録"])

    with tab_key:
        with st.form("mapping_by_label_form"):
            square_label = st.text_input(
                "Square照合キー",
                placeholder="例: パイ レモンクリーム / リエージュワッフル プレーン",
            )
            target_product = st.selectbox("アプリの商品", product_names, key="map_target_label")
            if st.form_submit_button("紐づけを保存", type="primary"):
                try:
                    save_product_mapping(square_label, target_product)
                    st.success(f"「{square_label.strip()}」→ {target_product}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with tab_parts:
        with st.form("mapping_by_parts_form"):
            sq_item = st.text_input("Square商品名", placeholder="例: パイ")
            sq_var = st.text_input("Squareバリエーション", placeholder="例: レモンクリーム（定価の場合は空でOK）")
            target_product2 = st.selectbox("アプリの商品", product_names, key="map_target_parts")
            preview = build_square_product_label(sq_item, sq_var)
            st.caption(f"照合キープレビュー: **{preview or '（未入力）'}**")
            if st.form_submit_button("紐づけを保存", type="primary"):
                try:
                    if not preview:
                        raise ValueError("Square商品名を入力してください。")
                    save_product_mapping(preview, target_product2)
                    st.success(f"「{preview}」→ {target_product2}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("#### CSVから未紐づけ行を探す")
    scan_file = st.file_uploader(
        "商品別マトリックスCSV（任意）",
        type=["csv"],
        key="mapping_scan_csv",
    )
    if scan_file:
        try:
            raw_df = read_uploaded_csv(scan_file)
            from square_csv import match_product_name

            sync_square_mappings()
            labels = list_square_labels_from_matrix(raw_df)
            need_map = [lb for lb in labels if not match_product_name(lb, product_names)]
            if not need_map:
                st.success("このCSVの行はすべて紐づけ済みです。")
            else:
                st.warning(f"未紐づけ {len(need_map)} 件 — 下で一括登録できます。")
                pick_label = st.selectbox("Square表記", need_map, key="quick_map_label")
                quick_target = st.selectbox("アプリ商品", product_names, key="quick_map_target")
                if st.button("この1件を紐づけ登録", type="primary"):
                    save_product_mapping(pick_label, quick_target)
                    st.success("登録しました。")
                    st.rerun()
        except Exception as exc:
            st.error(f"CSVの読み込みに失敗しました: {exc}")


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

    product_df = daily_df[daily_df["product_id"] == product_id]
    latest_date = get_latest_product_sales_date(daily_df, product_id) or get_latest_business_date(
        daily_df, day_totals
    )
    if latest_date is None:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    latest_row = product_df[product_df["date"] == to_ts(latest_date)]
    latest_units = int(latest_row["units_sold"].iloc[0]) if not latest_row.empty else 0
    day_total_sales = 0
    day_rows = daily_df[daily_df["date"] == to_ts(latest_date)]
    if not day_rows.empty:
        day_total_sales = int(day_rows["total_sales"].iloc[0])
    single_item_sales = calc_single_item_sales_indicator(day_total_sales, latest_units)

    lw_sales = last_week_same_day_sales(daily_df, product_id, latest_date)
    avg_4w = four_week_same_weekday_avg(daily_df, product_id, latest_date)

    latest_label = latest_date.strftime("%Y/%m/%d")
    st.caption(
        f"単品売上高（税抜） ＝ 店舗純売上（¥{day_total_sales:,}）÷ 販売数（{latest_units:,}個）"
        f" ＝ **¥{single_item_sales:,}**（{latest_label}）"
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"単品売上高・税抜（{latest_label}）", f"¥{single_item_sales:,}")
    with m2:
        st.metric(f"店舗純売上・税抜（{latest_label}）", f"¥{day_total_sales:,}")
    with m3:
        st.metric(f"販売数（{latest_label}）", f"{latest_units:,} 個")
    with m4:
        st.metric("前週同曜日販売数", f"{lw_sales:,} 個")

    recorded = list_recorded_dates(daily_df)
    data_min = recorded[-1] if recorded else latest_date
    data_max = latest_date

    st.markdown('<p class="section-title">詳細分析</p>', unsafe_allow_html=True)
    period_col, _ = st.columns([2, 3])
    with period_col:
        period_preset = st.selectbox(
            "集計期間（フード選択率・詳細分析の各タブ共通）",
            ["直近7日", "直近30日", "直近90日", "全期間", "日付を指定"],
            key="dashboard_avg_period_preset",
        )
    if period_preset == "日付を指定":
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            period_start = st.date_input(
                "開始日",
                value=max(data_min, latest_date - timedelta(days=29)),
                min_value=data_min,
                max_value=data_max,
                key="dashboard_avg_period_start",
            )
        with dcol2:
            period_end = st.date_input(
                "終了日",
                value=latest_date,
                min_value=data_min,
                max_value=data_max,
                key="dashboard_avg_period_end",
            )
    else:
        period_start, period_end = resolve_dashboard_period(
            period_preset,
            data_min=data_min,
            data_max=data_max,
            latest_date=latest_date,
        )
    period_label = f"{period_start:%Y/%m/%d} 〜 {period_end:%Y/%m/%d}"

    selection_rate = calc_food_selection_rate(
        daily_df, product_id, day_totals, period_start, period_end
    )
    sr1, sr2 = st.columns([1, 3])
    with sr1:
        st.metric(f"フード選択率（{period_label}）", f"{selection_rate:.2%}")
    with sr2:
        st.caption(
            f"**{selected_name}** の販売数 ÷ 店舗客数（指定期間の合計）。"
            " 発注予測タブの計算にもこの期間の選択率を使います。"
        )

    period_stats = calc_avg_units_sold_in_period(daily_df, product_id, period_start, period_end)
    weekday_units = calc_avg_units_by_weekday_in_period(daily_df, product_id, period_start, period_end)
    business_latest = get_latest_business_date(daily_df, day_totals) or latest_date
    latest_customers = 0
    latest_customers_row = day_totals[day_totals["date"] == to_ts(business_latest)]
    if not latest_customers_row.empty:
        latest_customers = int(latest_customers_row["total_customers"].iloc[0])
    business_label = business_latest.strftime("%Y/%m/%d")
    customer_stats = calc_avg_customers_in_period(day_totals, period_start, period_end)
    customer_trend = build_customer_trend_frame(day_totals, period_start, period_end)
    trend_df = build_product_trend_frame(
        daily_df, product_id, start_date=period_start, end_date=period_end
    )
    single_item_stats = calc_single_item_sales_stats_in_period(
        daily_df, product_id, period_start, period_end
    )
    sales_trend = trend_df[trend_df["units_sold"] > 0] if not trend_df.empty else trend_df

    tab_sales, tab_customers, tab_charts, tab_order = st.tabs(
        ["期間分析（販売数）", "店舗客数", "推移グラフ", "発注予測"]
    )

    with tab_sales:
        st.caption(f"集計期間: **{period_label}**（{period_stats['days']} 日分の登録データ）")
        ap1, ap2, ap3 = st.columns(3)
        with ap1:
            st.metric(f"平均販売個数", f"{period_stats['avg']:,.1f} 個/日")
        with ap2:
            st.metric("期間合計販売数", f"{period_stats['total']:,} 個")
        with ap3:
            st.metric("集計日数", f"{period_stats['days']:,} 日")
        if period_stats["active_days"] < period_stats["days"]:
            st.caption(
                f"販売があった日のみの平均: **{period_stats['total'] / period_stats['active_days']:,.1f} 個/日**"
                f"（{period_stats['active_days']} 日）"
                if period_stats["active_days"] > 0
                else "指定期間に販売実績がありません。"
            )

        st.markdown("##### 曜日別の平均販売個数")
        if weekday_units.empty:
            st.info("指定期間の販売データがないため、曜日別の平均を表示できません。")
        else:
            wk_chart, wk_table = st.columns([1.5, 1])
            with wk_chart:
                st.plotly_chart(
                    plot_weekday_avg_units(weekday_units, selected_name, period_label),
                    use_container_width=True,
                )
            with wk_table:
                table_view = weekday_units.copy()
                table_view["平均販売数"] = table_view["avg_units"].map(lambda v: f"{v:,.1f} 個")
                table_view["集計日数"] = table_view["days"].map(lambda v: f"{int(v)} 日")
                table_view["合計"] = table_view["total_units"].map(lambda v: f"{int(v):,} 個")
                st.dataframe(
                    table_view[["曜日", "平均販売数", "集計日数", "合計"]],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption("各曜日について、指定期間内に登録がある日の販売数の平均です。")

    with tab_customers:
        st.caption(f"集計期間: **{period_label}**")
        cp1, cp2, cp3, cp4 = st.columns(4)
        with cp1:
            st.metric(f"客数（{business_label}）", f"{latest_customers:,} 人")
        with cp2:
            st.metric("平均客数", f"{customer_stats['avg']:,.1f} 人/日")
        with cp3:
            st.metric("期間合計客数", f"{customer_stats['total']:,} 人")
        with cp4:
            st.metric("集計日数", f"{customer_stats['days']:,} 日")
        if customer_stats["active_days"] < customer_stats["days"] and customer_stats["active_days"] > 0:
            st.caption(
                f"客数が記録された日のみの平均: **{customer_stats['total'] / customer_stats['active_days']:,.1f} 人/日**"
                f"（{customer_stats['active_days']} 日）"
            )
        if customer_trend.empty:
            st.info("指定期間の客数データがありません。")
        else:
            st.plotly_chart(
                plot_customer_trend(customer_trend, customer_stats["avg"], period_label),
                use_container_width=True,
            )
            st.caption(f"破線は期間平均（**{customer_stats['avg']:,.1f} 人/日**）。")

    with tab_charts:
        st.caption(f"集計期間: **{period_label}**（販売数・単品売上高の推移）")
        si1, si2, si3 = st.columns(3)
        with si1:
            st.metric(
                f"平均単品売上高（{period_label}）",
                f"¥{single_item_stats['avg']:,.0f}",
            )
        with si2:
            st.metric("集計日数", f"{single_item_stats['days']:,} 日")
        with si3:
            st.metric("販売あり日数", f"{single_item_stats['active_days']:,} 日")

        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.plotly_chart(plot_weekday_comparison(lw_sales, avg_4w, selected_name), use_container_width=True)
            st.caption(f"過去1か月同曜日平均: **{avg_4w:,.1f} 個**")
        with c2:
            if not trend_df.empty:
                st.plotly_chart(
                    plot_daily_trend(trend_df, selected_name, period_label),
                    use_container_width=True,
                )
            else:
                st.info("指定期間の販売数データがありません。")

        st.markdown("##### 単品売上高の推移")
        if sales_trend.empty:
            st.info("指定期間に単品売上高を表示できるデータがありません（販売数0の日は除外）。")
        else:
            st.plotly_chart(
                plot_single_item_sales_trend(
                    sales_trend,
                    selected_name,
                    period_label,
                    float(single_item_stats["avg"]),
                ),
                use_container_width=True,
            )
            st.caption(
                "各日の **店舗純売上（税抜）÷ 当該商品の販売数**。破線は期間平均（販売あり日のみ）。"
            )

    with tab_order:
        st.caption(f"フード選択率（{period_label}）: **{selection_rate:.2%}** — 対象商品: {selected_name}")
        fc1, fc2, fc3 = st.columns(3)
        predicted_customers = fc1.number_input("予測客数", min_value=100, max_value=6000, value=2400, step=50)
        correction_pct = fc2.slider("天気・イベント補正（%）", -30, 50, 0, key="order_correction_pct")
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_daily_data_empty(daily_df):
        show_empty_data_notice()

    with st.sidebar:
        st.markdown("### データ管理")
        st.caption(f"登録日数: **{len(list_recorded_dates(daily_df))}** 日")
        if st.button("ゴミ日データを削除", type="secondary", use_container_width=True):
            n = purge_meaningless_days()
            if n:
                st.success(f"販売ゼロ・極小の日を {n} 日分削除しました。")
            else:
                st.info("削除対象の日はありませんでした。")
            st.rerun()
        if st.button("全データをリセット", type="secondary", use_container_width=True):
            reset_application_data()
            st.success("データを初期化しました。")
            st.rerun()

        st.divider()
        st.markdown("### 設定・修正")
        side_view = st.radio(
            "表示する画面",
            [
                "メイン（CSV・ダッシュボード）",
                "過去データの確認・修正",
                "商品・紐づけ設定",
            ],
            label_visibility="collapsed",
            key="sidebar_view",
        )

    if side_view == "過去データの確認・修正":
        render_history_tab(products_df, daily_df)
    elif side_view == "商品・紐づけ設定":
        render_products_mappings_tab(products_df)
    else:
        t_csv, t_dash = st.tabs(["Square CSVアップロード", "ダッシュボード・可視化"])
        with t_csv:
            render_square_upload_tab(products_df, daily_df)
        with t_dash:
            render_dashboard_tab(products_df, daily_df)


if __name__ == "__main__":
    main()
