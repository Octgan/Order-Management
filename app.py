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

try:
    from food_stock.logic import (
        apply_calendar_inputs,
        apply_planned_consumption,
        build_inventory_calendar_projection,
        build_inventory_projection,
        calendar_date_range,
        CALENDAR_PAST_DAYS,
        CALENDAR_TOTAL_DAYS,
        clear_actual_sales,
        clear_consumption_plan,
        days_until_stockout,
        get_delivery_weekdays,
        init_actual_sales_csv,
        init_consumption_plan_csv,
        init_deliveries_csv,
        init_delivery_plan_csv,
        init_inventory_csv,
        load_inventory_df,
        load_manual_actual_sales,
        load_manual_planned_use,
        replace_actual_sales_for_period,
        replace_delivery_plan_for_period,
        save_consumption_plan,
        save_delivery_weekdays,
        save_product_inventory,
        sync_inventory_products,
        weekday_labels_text,
    )
except ImportError as _food_stock_import_error:
    st.error(f"在庫モジュールの読み込みに失敗しました: {_food_stock_import_error}")
    st.stop()
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
import shared_storage as store

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
APP_TITLE = "Square CSV連携 フード発注管理"
COLD_STORAGE_CAPACITY = 300
DATA_VERSION = 5  # 仕様変更時に上げると data/ を初期化（4→5は税抜へ戻す）
STANDARD_CONSUMPTION_TAX_RATE = 0.10

DATA_DIR = store.DATA_DIR
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

MAIN_TAB_OPTIONS = [
    "Square CSVアップロード",
    "ダッシュボード・可視化",
    "在庫管理",
    "MTGレポート",
]

DASHBOARD_DETAIL_TABS = [
    "期間分析（販売数）",
    "店舗客数",
    "推移グラフ",
    "発注予測",
]

_PLOTLY_CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def clear_data_caches() -> None:
    _cached_read_products.clear()
    _cached_read_daily_sales.clear()
    _cached_read_mappings.clear()


@st.cache_data(show_spinner=False)
def _cached_read_products(_mtime: float) -> pd.DataFrame:
    df = store.read_csv(PRODUCTS_CSV)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(1).astype(int)
    return df


@st.cache_data(show_spinner=False)
def _cached_read_daily_sales(_mtime: float) -> pd.DataFrame:
    if not DAILY_CSV.exists() or DAILY_CSV.stat().st_size == 0:
        return pd.DataFrame()
    df = store.read_csv(DAILY_CSV)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    for col in ["total_sales", "total_customers", "unit_price", "units_sold", "product_sales"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def _cached_read_mappings(_mtime: float) -> pd.DataFrame:
    if not MAPPINGS_CSV.exists() or MAPPINGS_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    return store.read_csv(MAPPINGS_CSV)


def stock_form_revision() -> int:
    return int(st.session_state.get("inv_stock_form_rev", 0))


def bump_stock_form_revision() -> None:
    st.session_state["inv_stock_form_rev"] = stock_form_revision() + 1


def stock_input_widget_keys(product_id: str, rev: int | None = None) -> tuple[str, str, str]:
    rev_val = stock_form_revision() if rev is None else rev
    pid = str(product_id)
    return (
        f"all_inv_stock_{pid}_{rev_val}",
        f"all_inv_safety_{pid}_{rev_val}",
        f"all_inv_max_{pid}_{rev_val}",
    )


def reset_inventory_input_widgets(
    product_list: pd.DataFrame | None = None,
    *,
    product_id: str | None = None,
) -> None:
    """在庫保存後に入力ウィジェットを初期化する。"""
    if product_list is not None:
        bump_stock_form_revision()
    if product_id is not None:
        rev_key = f"order_stock_rev_{product_id}"
        st.session_state[rev_key] = int(st.session_state.get(rev_key, 0)) + 1


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
        /* 在庫カレンダー: 実売数・計画消費・納品数（入力列）を強調 */
        [data-testid="stDataEditor"] [aria-colindex="5"],
        [data-testid="stDataEditor"] [aria-colindex="6"],
        [data-testid="stDataEditor"] [aria-colindex="7"] {
            background-color: #fff9f2 !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="5"][role="columnheader"] {
            background-color: #e8f8ef !important;
            font-weight: 700 !important;
            color: #1e6b3a !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="6"][role="columnheader"] {
            background-color: #ffe8ec !important;
            font-weight: 700 !important;
            color: #c0392b !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="7"][role="columnheader"] {
            background-color: #e8f4ff !important;
            font-weight: 700 !important;
            color: #1a5276 !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="5"] input {
            border: 2px solid #27ae60 !important;
            border-radius: 6px !important;
            background-color: #fff !important;
            font-weight: 600 !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="6"] input {
            border: 2px solid #e94560 !important;
            border-radius: 6px !important;
            background-color: #fff !important;
            font-weight: 600 !important;
        }
        [data-testid="stDataEditor"] [aria-colindex="7"] input {
            border: 2px solid #2980b9 !important;
            border-radius: 6px !important;
            background-color: #fff !important;
            font-weight: 600 !important;
        }
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
    store.write_csv(products_df, PRODUCTS_CSV)

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
        store.write_csv(daily_df, DAILY_CSV)

    store.write_text(VERSION_FILE, str(DATA_VERSION))


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
    store.write_csv(pd.DataFrame(columns=DAILY_SALES_COLUMNS), DAILY_CSV)


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
    store.write_csv(pd.DataFrame(rows), PRODUCTS_CSV)


def reset_application_data() -> None:
    """営業データと商品マスタを初期状態に戻す（紐づけ設定は保持）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_default_products()
    init_empty_daily_csv()
    init_inventory_csv(load_products())
    init_deliveries_csv()
    init_consumption_plan_csv()
    init_actual_sales_csv()
    if not MAPPINGS_CSV.exists():
        init_empty_mappings_csv()
    store.write_text(VERSION_FILE, str(DATA_VERSION))
    sync_square_mappings()


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stored_version = 0
    if VERSION_FILE.exists():
        try:
            stored_version = int(store.read_text(VERSION_FILE).strip())
        except ValueError:
            stored_version = 0

    if stored_version < DATA_VERSION:
        if stored_version == 4 and DATA_VERSION >= 5:
            migrate_v4_to_tax_exclusive()
        elif stored_version == 3 and DATA_VERSION >= 5:
            store.write_text(VERSION_FILE, str(DATA_VERSION))
        else:
            reset_application_data()
        return

    if not PRODUCTS_CSV.exists():
        write_default_products()
    if not DAILY_CSV.exists():
        init_empty_daily_csv()
    if not MAPPINGS_CSV.exists():
        init_empty_mappings_csv()
    init_inventory_csv(load_products())
    init_deliveries_csv()
    init_consumption_plan_csv()
    init_actual_sales_csv()
    sync_square_mappings()


def load_products() -> pd.DataFrame:
    return _cached_read_products(_file_mtime(PRODUCTS_CSV))


def active_products(products_df: pd.DataFrame) -> pd.DataFrame:
    active = products_df[products_df["is_active"] == 1].copy()
    return active if not active.empty else products_df.copy()


def init_empty_mappings_csv() -> None:
    store.write_csv(pd.DataFrame(columns=MAPPING_COLUMNS), MAPPINGS_CSV)


def load_product_mappings_df() -> pd.DataFrame:
    return _cached_read_mappings(_file_mtime(MAPPINGS_CSV))


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
    store.write_csv(df, MAPPINGS_CSV)
    sync_square_mappings()


def delete_product_mapping(square_label: str) -> None:
    label = square_label.strip()
    df = load_product_mappings_df()
    df = df[df["square_label"].astype(str).str.strip() != label]
    if df.empty:
        init_empty_mappings_csv()
    else:
        store.write_csv(df, MAPPINGS_CSV)
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
    store.write_csv(products_df, PRODUCTS_CSV)
    sync_inventory_products(products_df)


def update_product_price(product_id: str, unit_price: int) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "unit_price"] = int(unit_price)
    store.write_csv(products_df, PRODUCTS_CSV)


def deactivate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 0
    store.write_csv(products_df, PRODUCTS_CSV)


def activate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 1
    store.write_csv(products_df, PRODUCTS_CSV)


def load_daily_sales() -> pd.DataFrame:
    return _cached_read_daily_sales(_file_mtime(DAILY_CSV))


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
    store.write_csv(merged, DAILY_CSV)


def delete_daily_record(target_date: date) -> None:
    daily_df = load_daily_sales()
    if daily_df.empty:
        return
    daily_df = daily_df[daily_df["date"] != to_ts(target_date)]
    if daily_df.empty:
        init_empty_daily_csv()
    else:
        store.write_csv(daily_df, DAILY_CSV)


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
        store.write_csv(pd.concat(keep_parts, ignore_index=True), DAILY_CSV)
    else:
        init_empty_daily_csv()
    return removed


def bulk_import_days(imports: list[DayImport], products_df: pd.DataFrame, overwrite: bool = True) -> tuple[int, int, bool]:
    """CSV結合結果を daily_sales.csv に保存（日付単位で上書き）。"""
    daily_df = load_daily_sales()
    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    created, updated = 0, 0
    product_list = active_products(products_df)
    product_names = product_list["name"].astype(str).tolist()
    import_dates: set[date] = set()
    new_rows: list[dict[str, Any]] = []
    now = datetime.now().isoformat()

    for day_data in imports:
        if day_data.day in existing_dates:
            if not overwrite:
                continue
            updated += 1
        else:
            created += 1
        import_dates.add(day_data.day)
        units = {name: day_data.units_by_product.get(name, 0) for name in product_names}
        sales: dict[str, int] = {}
        if day_data.product_sales_by_product:
            sales = {name: day_data.product_sales_by_product.get(name, 0) for name in product_names}
        if not sales:
            sales = allocate_product_sales_from_store(day_data.total_sales, units)
        target_ts = to_ts(day_data.day)
        for _, row in product_list.iterrows():
            name = str(row["name"])
            new_rows.append(
                {
                    "date": target_ts.strftime("%Y-%m-%d"),
                    "total_sales": int(day_data.total_sales),
                    "total_customers": int(day_data.total_customers),
                    "product_id": row["product_id"],
                    "product_name": name,
                    "unit_price": int(row["unit_price"]),
                    "units_sold": int(units.get(name, 0)),
                    "product_sales": int(sales.get(name, 0)),
                    "created_at": now,
                }
            )

    if not new_rows:
        return 0, 0, True

    if not daily_df.empty:
        daily_df = daily_df[~daily_df["date"].dt.date.isin(import_dates)]
    merged = pd.concat([daily_df, pd.DataFrame(new_rows)], ignore_index=True)
    cloud_ok = store.write_csv(merged, DAILY_CSV)
    clear_data_caches()
    return created, updated, cloud_ok


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


def calc_food_selection_period_totals(
    daily_df: pd.DataFrame,
    product_id: str,
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[int, int]:
    """指定期間の対象商品販売数合計と店舗客数合計。"""
    if daily_df.empty or day_totals.empty or start_date > end_date:
        return 0, 0
    start = to_ts(start_date)
    end = to_ts(end_date)
    sub = daily_df[
        (daily_df["product_id"] == product_id)
        & (daily_df["date"] >= start)
        & (daily_df["date"] <= end)
    ]
    totals = day_totals[(day_totals["date"] >= start) & (day_totals["date"] <= end)]
    units = int(sub["units_sold"].sum()) if not sub.empty else 0
    customers = int(totals["total_customers"].sum())
    return units, customers


def calc_food_selection_rate(
    daily_df: pd.DataFrame,
    product_id: str,
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> float:
    """指定期間のフード選択率 = 対象商品の販売数合計 ÷ 店舗客数合計。"""
    units, customers = calc_food_selection_period_totals(
        daily_df, product_id, day_totals, start_date, end_date
    )
    if customers == 0:
        return 0.0
    return float(units / customers)


def calc_all_foods_selection_period_totals(
    daily_df: pd.DataFrame,
    product_ids: list[str],
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[int, int]:
    """全フードの販売数合計と店舗客数合計。"""
    if daily_df.empty or day_totals.empty or start_date > end_date or not product_ids:
        return 0, 0
    start = to_ts(start_date)
    end = to_ts(end_date)
    pids = {str(p) for p in product_ids}
    sub = daily_df[
        daily_df["product_id"].astype(str).isin(pids)
        & (daily_df["date"] >= start)
        & (daily_df["date"] <= end)
    ]
    totals = day_totals[(day_totals["date"] >= start) & (day_totals["date"] <= end)]
    units = int(sub["units_sold"].sum()) if not sub.empty else 0
    customers = int(totals["total_customers"].sum())
    return units, customers


def build_food_units_breakdown_in_period(
    daily_df: pd.DataFrame,
    products_df: pd.DataFrame,
    day_totals: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, int, int]:
    """フード別の期間販売数・選択率（全フード合計の内訳）。"""
    product_list = active_products(products_df)
    ids = product_list["product_id"].astype(str).tolist()
    total_units, customers = calc_all_foods_selection_period_totals(
        daily_df, ids, day_totals, start_date, end_date
    )
    rows: list[dict[str, Any]] = []
    for _, prow in product_list.iterrows():
        pid = str(prow["product_id"])
        pname = str(prow["name"])
        units, _ = calc_food_selection_period_totals(
            daily_df, pid, day_totals, start_date, end_date
        )
        rate = float(units / customers) if customers > 0 else 0.0
        rows.append(
            {
                "product_name": pname,
                "product_id": pid,
                "units": int(units),
                "selection_rate": rate,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("units", ascending=False)
    return df, total_units, customers


FOOD_PIE_COLORS = [
    "#e94560",
    "#0f3460",
    "#16a085",
    "#f39c12",
    "#8e44ad",
    "#2980b9",
    "#27ae60",
    "#c0392b",
    "#d35400",
    "#7f8c8d",
]


def plot_all_foods_selection_rate_pie(
    breakdown_df: pd.DataFrame,
    total_customers: int,
    period_label: str,
    total_units: int,
    total_rate: float,
) -> go.Figure:
    """全フード合計のフード選択率（商品別内訳）ドーナツグラフ。"""
    fig = go.Figure()
    center_text = "—"
    if total_customers <= 0:
        fig.add_trace(
            go.Pie(
                labels=["データなし"],
                values=[1],
                hole=0.5,
                marker=dict(colors=["#e0e0e0"]),
                textinfo="label",
                hoverinfo="skip",
            )
        )
    else:
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        hover_units: list[int] = []
        for _, row in breakdown_df.iterrows():
            units = int(row["units"])
            if units <= 0:
                continue
            pct = units / total_customers * 100
            labels.append(str(row["product_name"]))
            values.append(pct)
            colors.append(FOOD_PIE_COLORS[len(colors) % len(FOOD_PIE_COLORS)])
            hover_units.append(units)
        food_pct_sum = sum(values)
        other_pct = max(0.0, 100.0 - food_pct_sum)
        if other_pct > 0.05:
            labels.append("その他（客数ベース）")
            values.append(other_pct)
            colors.append("#ecf0f1")
            hover_units.append(0)
        if not values:
            labels = ["フード販売なし", "その他（客数ベース）"]
            values = [0.001, 99.999]
            colors = ["#e94560", "#ecf0f1"]
            hover_units = [0, 0]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors),
                textinfo="percent",
                textposition="inside",
                hovertemplate="%{label}<br>%{percent}<br>販売: %{customdata:,} 個<extra></extra>",
                customdata=hover_units,
            )
        )
        if total_rate >= 1.0:
            center_text = f"{total_rate:.1%}\n(1人あたり1個超)"
        else:
            center_text = f"{total_rate:.1%}"

    fig.update_layout(
        title=f"全フード選択率（{period_label}）",
        template="plotly_white",
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(t=48, b=80, l=16, r=16),
        annotations=[
            dict(
                text=center_text,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18, color="#1a1a2e"),
            )
        ],
    )
    return fig


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
_PENDING_CSV_KEY = "pending_csv_import_payload"
_CSV_KIND_LABELS = {
    "product_matrix": "商品別マトリックス",
    "product_period_summary": "商品売上サマリー（期間）",
    "sales_summary": "売上サマリー",
    "unknown": "未判定",
}


def _serialize_day_import(day_data: DayImport) -> dict[str, Any]:
    return {
        "day": day_data.day.isoformat(),
        "total_sales": int(day_data.total_sales),
        "total_customers": int(day_data.total_customers),
        "units_by_product": dict(day_data.units_by_product),
        "product_sales_by_product": dict(day_data.product_sales_by_product or {}),
    }


def _deserialize_day_import(payload: dict[str, Any]) -> DayImport:
    return DayImport(
        date.fromisoformat(str(payload["day"])),
        int(payload["total_sales"]),
        int(payload["total_customers"]),
        dict(payload["units_by_product"]),
        dict(payload.get("product_sales_by_product") or {}) or None,
    )


def _clear_pending_csv_payload() -> None:
    st.session_state.pop(_PENDING_CSV_KEY, None)


def render_square_upload_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">Square売上CSVアップロード（ルートB）</p>', unsafe_allow_html=True)
    st.caption(
        "2枚のCSVを同時にアップロードしてください。"
        " ①商品別マトリックス、または期間の「商品売上サマリー」"
        " ②売上サマリー（客数・店舗純売上＝税抜で取り込み）"
        " — プレビュー後 **「データを一括取り込み」** を押すと保存されます。"
    )

    uploaded_list = st.file_uploader(
        "Square CSVファイル（最大2枚）",
        type=["csv"],
        accept_multiple_files=True,
    )
    overwrite = st.checkbox("既存日付のデータは上書きする", value=True)

    if uploaded_list:
        files = uploaded_list if isinstance(uploaded_list, list) else [uploaded_list]
        if len(files) > 2:
            st.error("アップロードは2ファイルまでにしてください。")
            _clear_pending_csv_payload()
            return
        try:
            imports, warnings = parse_dual_csv_upload(files, products_df)
        except Exception as exc:
            st.error(f"CSVの読み込みに失敗しました: {exc}")
            _clear_pending_csv_payload()
            return
        if not imports:
            st.warning("取り込み可能な日次データがありませんでした。")
            _clear_pending_csv_payload()
            return

        file_labels: list[dict[str, str]] = []
        mapping_records: list[dict[str, Any]] = []
        for f in files:
            kind, product_df_raw = load_uploaded_dataframe(f, products_df)
            file_labels.append({"name": f.name, "kind": kind})
            if kind in ("product_matrix", "product_period_summary"):
                mapping_df = summarize_square_row_mapping(product_df_raw, products_df)
                if not mapping_df.empty:
                    mapping_records.extend(mapping_df.to_dict("records"))

        st.session_state[_PENDING_CSV_KEY] = {
            "imports": [_serialize_day_import(d) for d in imports],
            "warnings": warnings,
            "file_labels": file_labels,
            "mapping_records": mapping_records,
        }

    payload = st.session_state.get(_PENDING_CSV_KEY)
    if not payload:
        if is_daily_data_empty(daily_df):
            st.info(EMPTY_DATA_MESSAGE)
            st.markdown(
                "1. **商品別**の横持ちCSV（登録フード品目の日別販売数・売上）  \n"
                "2. **売上サマリー**のCSV（日別の総客数・店舗総売上）  \n"
                "3. 2枚をまとめてドラッグ＆ドロップ → プレビュー確認 → **「データを一括取り込み」** を押す"
            )
        else:
            st.info("CSVをアップロードすると、取り込みプレビューが表示されます。")
        return

    imports = [_deserialize_day_import(item) for item in payload["imports"]]
    warnings = list(payload.get("warnings") or [])
    file_labels = list(payload.get("file_labels") or [])
    mapping_records = list(payload.get("mapping_records") or [])

    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    overlap = [d.day for d in imports if d.day in existing_dates]
    if overlap:
        st.warning(f"既存データと重複する日付: {len(overlap)}日（上書き: {'ON' if overwrite else 'OFF'}）")

    st.info(
        f"取り込み待ち: **{len(imports)}** 日分のプレビューがあります。"
        " 内容を確認したら下のボタンで保存してください。"
    )
    import_ready = st.button("データを一括取り込み", type="primary", use_container_width=True)
    if import_ready:
        created, updated, cloud_ok = bulk_import_days(imports, products_df, overwrite=overwrite)
        if created == 0 and updated == 0:
            st.warning("保存された日がありません。上書きOFFで重複日のみの場合は取り込まれません。")
        else:
            _clear_pending_csv_payload()
            total_days = created + updated
            if store.is_cloud_enabled() and cloud_ok:
                store.set_persist_notice(
                    f"取り込み完了: {total_days} 日分を保存し、クラウドにも同期しました。",
                    "success",
                )
            elif store.is_cloud_enabled():
                store.set_persist_notice(
                    f"取り込み完了: {total_days} 日分をこの端末に保存しましたが、クラウド同期に失敗しました。",
                    "warning",
                )
            else:
                store.set_persist_notice(
                    f"取り込み完了: {total_days} 日分を保存しました。",
                    "success",
                )
            st.rerun()

    st.markdown("#### アップロードされたファイル")
    for item in file_labels:
        label = _CSV_KIND_LABELS.get(item.get("kind", ""), item.get("kind", "未判定"))
        st.write(f"- **{item.get('name', '')}** → {label}")

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

    if mapping_records:
        st.markdown("#### 商品名の紐づけ確認（Square → アプリ）")
        st.caption(
            "「パイ」などはバリエーション列（ミート／レモンクリーム）と組み合わせて判定します。"
            " 未紐づけがある場合は取り込み前にご確認ください。"
        )
        st.dataframe(pd.DataFrame(mapping_records), use_container_width=True, hide_index=True)

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
        "新メニューを追加すると、CSV取り込み・ダッシュボード・**在庫管理**の対象に含められます。"
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

    food_breakdown, all_foods_units, period_customers = build_food_units_breakdown_in_period(
        daily_df, products_df, day_totals, period_start, period_end
    )
    all_foods_rate = (
        float(all_foods_units / period_customers) if period_customers > 0 else 0.0
    )
    product_selection_rate = calc_food_selection_rate(
        daily_df, product_id, day_totals, period_start, period_end
    )
    sr1, sr2 = st.columns([1, 2])
    with sr1:
        st.metric(f"全フード選択率（{period_label}）", f"{all_foods_rate:.2%}")
        st.caption(
            f"全フード販売 **{all_foods_units:,}** 個 ÷ 客数 **{period_customers:,}** 人"
        )
        st.caption(f"選択中商品（{selected_name}）: **{product_selection_rate:.2%}**")
    with sr2:
        st.plotly_chart(
            plot_all_foods_selection_rate_pie(
                food_breakdown,
                period_customers,
                period_label,
                all_foods_units,
                all_foods_rate,
            ),
            use_container_width=True,
            config=_PLOTLY_CHART_CONFIG,
        )
    if not food_breakdown.empty:
        bd_view = food_breakdown.copy()
        bd_view["選択率"] = bd_view["selection_rate"].map(lambda r: f"{r:.2%}")
        bd_view["販売数"] = bd_view["units"].map(lambda u: f"{int(u):,} 個")
        st.dataframe(
            bd_view[["product_name", "販売数", "選択率"]].rename(
                columns={"product_name": "フード商品"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "丸グラフは **全フード** の販売構成（各商品の販売数 ÷ 店舗客数）。"
        " 発注予測タブでは **選択中の商品** の選択率を使います。"
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

    detail_tab = st.radio(
        "詳細分析タブ",
        DASHBOARD_DETAIL_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_detail_tab",
    )

    if detail_tab == DASHBOARD_DETAIL_TABS[0]:
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
                    config=_PLOTLY_CHART_CONFIG,
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

    elif detail_tab == DASHBOARD_DETAIL_TABS[1]:
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
                config=_PLOTLY_CHART_CONFIG,
            )
            st.caption(f"破線は期間平均（**{customer_stats['avg']:,.1f} 人/日**）。")

    elif detail_tab == DASHBOARD_DETAIL_TABS[2]:
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
            st.plotly_chart(
                plot_weekday_comparison(lw_sales, avg_4w, selected_name),
                use_container_width=True,
                config=_PLOTLY_CHART_CONFIG,
            )
            st.caption(f"過去1か月同曜日平均: **{avg_4w:,.1f} 個**")
        with c2:
            if not trend_df.empty:
                st.plotly_chart(
                    plot_daily_trend(trend_df, selected_name, period_label),
                    use_container_width=True,
                    config=_PLOTLY_CHART_CONFIG,
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
                config=_PLOTLY_CHART_CONFIG,
            )
            st.caption(
                "各日の **店舗純売上（税抜）÷ 当該商品の販売数**。破線は期間平均（販売あり日のみ）。"
            )

    elif detail_tab == DASHBOARD_DETAIL_TABS[3]:
        init_inventory_csv(products_df)
        sync_inventory_products(products_df)
        order_inv_df = load_inventory_df()
        order_default_stock, order_default_safety, order_default_max = get_inventory_row_defaults(
            order_inv_df, str(product_id)
        )

        st.caption(
            f"フード選択率（{period_label}）: **{product_selection_rate:.2%}** — 対象商品: {selected_name}"
            f"（全フード合計: {all_foods_rate:.2%}）"
        )
        fc1, fc2, fc3 = st.columns(3)
        predicted_customers = fc1.number_input(
            "予測客数", min_value=100, max_value=6000, value=2400, step=50, key="order_pred_customers"
        )
        correction_pct = fc2.slider("天気・イベント補正（%）", -30, 50, 0, key="order_correction_pct")

        order_stock_rev = int(st.session_state.get(f"order_stock_rev_{product_id}", 0))
        with fc3:
            with st.form(f"order_stock_form_{product_id}_{order_stock_rev}", clear_on_submit=False):
                current_stock = st.number_input(
                    f"現在の在庫（{selected_name}）",
                    min_value=0,
                    max_value=9999,
                    value=order_default_stock,
                    step=1,
                )
                save_stock = st.form_submit_button("在庫を登録", use_container_width=True)

        if save_stock:
            cloud_ok = save_product_inventory(
                str(product_id),
                selected_name,
                int(current_stock),
                int(order_default_safety),
                int(order_default_max),
            )
            reset_inventory_input_widgets(product_id=str(product_id))
            st.session_state["inv_data_rev"] = int(st.session_state.get("inv_data_rev", 0)) + 1
            if store.is_cloud_enabled() and cloud_ok:
                store.set_persist_notice(
                    f"{selected_name} の在庫を保存し、クラウドにも同期しました。",
                    "success",
                )
            elif store.is_cloud_enabled():
                store.set_persist_notice(
                    f"{selected_name} の在庫はこの端末に保存しましたが、クラウド同期に失敗しました。",
                    "warning",
                )
            else:
                store.set_persist_notice(f"{selected_name} の在庫を保存しました。", "success")
            st.rerun()

        os_caption = st.columns([1, 3])
        with os_caption[0]:
            st.write("")
        with os_caption[1]:
            st.caption(
                "在庫は商品ごとに保存されます。"
                " **在庫管理** タブの一覧からまとめて登録することもできます。"
            )

        correction_factor = 1.0 + correction_pct / 100.0
        recommended = max(
            0,
            int(
                round(
                    predicted_customers * product_selection_rate * correction_factor - current_stock
                )
            ),
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("補正係数", f"×{correction_factor:.2f}")
        with r2:
            st.metric(
                "理論需要数",
                f"{int(predicted_customers * product_selection_rate * correction_factor):,} 個",
            )
        with r3:
            st.metric("推奨発注量", f"{recommended:,} 個")

        st.code(
            f"推奨発注量 = (予測客数 {predicted_customers:,} × フード選択率 {product_selection_rate:.4f}) "
            f"× 補正係数 {correction_factor:.2f} − 現在在庫 {current_stock}\n"
            f"         = {recommended:,} 個",
            language="text",
        )

        projected_total = int(current_stock) + recommended
        if projected_total > order_default_max:
            st.error(
                f"発注後の在庫（{projected_total:,} 個）が収納MAX（{order_default_max:,} 個）を超えます。"
            )
        elif projected_total > order_default_max * 0.85:
            st.warning(
                f"発注後の在庫が収納MAX（{order_default_max:,} 個）の85%を超えています。"
            )
        else:
            st.success(f"収納MAX（{order_default_max:,} 個）内の推奨発注量です。")


def _load_mtg_report_modules() -> tuple[Any, ...]:
    """MTG PDF 用モジュール（起動時ではなくタブ表示時に読み込む）。"""
    from mtg_report import (
        FoodReportSection,
        MtgReportContext,
        build_mtg_report_pdf,
        plotly_fig_to_png,
    )

    return FoodReportSection, MtgReportContext, build_mtg_report_pdf, plotly_fig_to_png


def render_mtg_report_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    """MTG 用 PDF レポートの作成・ダウンロード（全フード）。"""
    try:
        FoodReportSection, MtgReportContext, build_mtg_report_pdf, plotly_fig_to_png = (
            _load_mtg_report_modules()
        )
    except ImportError as exc:
        st.error(f"MTGレポート機能を読み込めません: {exc}")
        st.caption("ダッシュボード・在庫管理は引き続き利用できます。")
        return

    st.markdown('<p class="section-title">MTGレポート（PDF）</p>', unsafe_allow_html=True)
    st.caption(
        "**全フード** の売上・選択率・発注予測・在庫見込みを1つの PDF にまとめます。"
        " 会議の資料としてダウンロードしてご利用ください。"
    )
    if daily_df.empty:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    daily_df = daily_df.sort_values("date")
    day_totals = get_day_totals(daily_df)
    product_list = active_products(products_df)
    business_latest = get_latest_business_date(daily_df, day_totals)
    if business_latest is None:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    recorded = list_recorded_dates(daily_df)
    data_min = recorded[-1] if recorded else business_latest
    data_max = business_latest

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        period_preset = st.selectbox(
            "集計期間",
            ["直近7日", "直近30日", "直近90日", "全期間", "日付を指定"],
            key="mtg_period_preset",
        )
    with pcol2:
        include_inventory = st.checkbox("在庫見込みを含める", value=True, key="mtg_include_inv")

    if period_preset == "日付を指定":
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            period_start = st.date_input(
                "開始日",
                value=max(data_min, business_latest - timedelta(days=29)),
                min_value=data_min,
                max_value=data_max,
                key="mtg_period_start",
            )
        with dcol2:
            period_end = st.date_input(
                "終了日",
                value=business_latest,
                min_value=data_min,
                max_value=data_max,
                key="mtg_period_end",
            )
    else:
        period_start, period_end = resolve_dashboard_period(
            period_preset,
            data_min=data_min,
            data_max=data_max,
            latest_date=business_latest,
        )
    period_label = f"{period_start:%Y/%m/%d} 〜 {period_end:%Y/%m/%d}"

    st.markdown("##### 発注予測の前提（全フード共通・PDFに記載）")
    fc1, fc2 = st.columns(2)
    with fc1:
        predicted_customers = st.number_input(
            "予測客数", min_value=100, max_value=6000, value=2400, step=50, key="mtg_pred_customers"
        )
    with fc2:
        correction_pct = st.slider(
            "天気・イベント補正（%）", -30, 50, 0, key="mtg_correction_pct"
        )

    inv_horizon = 14
    if include_inventory:
        inv_horizon = st.selectbox(
            "在庫見込みの日数",
            [7, 14, 21, 30],
            index=1,
            format_func=lambda d: f"{d}日間",
            key="mtg_inv_horizon",
        )

    report_memo = st.text_area(
        "MTGメモ（任意・PDFに記載）",
        placeholder="例: 来週は雨予報のため補正 -10% で検討",
        key="mtg_memo",
    )

    include_charts = st.checkbox("グラフをPDFに含める", value=True, key="mtg_include_charts")
    st.caption(f"対象フード: **{len(product_list)}** 商品（在庫は各商品の登録値を使用）")

    if st.button("PDFを作成", type="primary", key="mtg_build_pdf"):
        day_rows = daily_df[daily_df["date"] == to_ts(business_latest)]
        day_total_sales = int(day_rows["total_sales"].iloc[0]) if not day_rows.empty else 0

        food_breakdown, all_foods_units, period_customers = build_food_units_breakdown_in_period(
            daily_df, products_df, day_totals, period_start, period_end
        )
        all_foods_rate = (
            float(all_foods_units / period_customers) if period_customers > 0 else 0.0
        )
        customer_stats = calc_avg_customers_in_period(day_totals, period_start, period_end)
        customer_trend = build_customer_trend_frame(day_totals, period_start, period_end)
        correction_factor = 1.0 + correction_pct / 100.0

        chart_images: dict[str, bytes] = {}
        if include_charts:
            chart_images["selection_pie"] = plotly_fig_to_png(
                plot_all_foods_selection_rate_pie(
                    food_breakdown,
                    period_customers,
                    period_label,
                    all_foods_units,
                    all_foods_rate,
                ),
                height=400,
            )
            if not customer_trend.empty:
                chart_images["customer_trend"] = plotly_fig_to_png(
                    plot_customer_trend(customer_trend, customer_stats["avg"], period_label),
                    height=360,
                )
            chart_images = {k: v for k, v in chart_images.items() if v}

        inv_df = load_inventory_df()
        food_sections: list[FoodReportSection] = []
        for _, prow in product_list.iterrows():
            pid = str(prow["product_id"])
            pname = str(prow["name"])
            latest_date = (
                get_latest_product_sales_date(daily_df, pid) or business_latest
            )
            product_df = daily_df[daily_df["product_id"] == pid]
            latest_row = product_df[product_df["date"] == to_ts(latest_date)]
            latest_units = int(latest_row["units_sold"].iloc[0]) if not latest_row.empty else 0
            day_sales_row = daily_df[daily_df["date"] == to_ts(latest_date)]
            day_sales = int(day_sales_row["total_sales"].iloc[0]) if not day_sales_row.empty else 0
            single_item_sales = calc_single_item_sales_indicator(day_sales, latest_units)

            period_units, _ = calc_food_selection_period_totals(
                daily_df, pid, day_totals, period_start, period_end
            )
            selection_rate = (
                float(period_units / period_customers) if period_customers > 0 else 0.0
            )
            inv_row = inv_df[inv_df["product_id"].astype(str) == pid]
            inv_stock = int(inv_row["current_stock"].iloc[0]) if not inv_row.empty else 0
            recommended = max(
                0,
                int(
                    round(
                        predicted_customers * selection_rate * correction_factor - inv_stock
                    )
                ),
            )

            inventory_summary: dict[str, Any] | None = None
            if include_inventory:
                inv_safety = int(inv_row["safety_stock"].iloc[0]) if not inv_row.empty else 10
                inv_max = (
                    int(inv_row["max_stock"].iloc[0])
                    if not inv_row.empty and "max_stock" in inv_row.columns
                    else COLD_STORAGE_CAPACITY
                )
                if inv_max <= 0:
                    inv_max = COLD_STORAGE_CAPACITY
                start_date = date.today()
                projection = build_inventory_projection(
                    daily_df,
                    pid,
                    inv_stock,
                    start_date,
                    int(inv_horizon),
                    None,
                    safety_stock=inv_safety,
                    max_stock=inv_max,
                )
                end_date = start_date + timedelta(days=int(inv_horizon) - 1)
                manual_use = load_manual_planned_use(pid, start_date, end_date)
                actual_use = load_manual_actual_sales(pid, start_date, end_date)
                projection = apply_calendar_inputs(
                    projection,
                    manual_use,
                    actual_use,
                    today=start_date,
                    opening_stock_today=inv_stock,
                    safety_stock=inv_safety,
                    max_stock=inv_max,
                )
                stockout_in = days_until_stockout(projection, from_date=start_date)
                stockout_label = (
                    f"{stockout_in} 日後" if stockout_in is not None else "なし（期間内は維持）"
                )
                inventory_summary = {
                    "safety_stock": inv_safety,
                    "max_stock": inv_max,
                    "horizon": inv_horizon,
                    "avg_use": float(
                        projection["effective_use"].mean()
                        if "effective_use" in projection.columns
                        else projection["planned_use"].mean()
                    )
                    if not projection.empty
                    else 0.0,
                    "min_stock": int(projection["stock_end"].min())
                    if not projection.empty
                    else 0,
                    "end_stock": int(projection["stock_end"].iloc[-1])
                    if not projection.empty
                    else 0,
                    "stockout_label": stockout_label,
                }
                if include_charts and not projection.empty:
                    chart_images[f"inventory_{pid}"] = plotly_fig_to_png(
                        plot_inventory_vertical_calendar(projection, pname, inv_max),
                        width=700,
                        height=max(420, 28 * len(projection)),
                    )

            food_sections.append(
                FoodReportSection(
                    product_name=pname,
                    product_id=pid,
                    selection_rate=selection_rate,
                    period_units=period_units,
                    latest_label=latest_date.strftime("%Y/%m/%d"),
                    latest_units=latest_units,
                    latest_single_item_sales=single_item_sales,
                    lw_sales=last_week_same_day_sales(daily_df, pid, latest_date),
                    avg_4w=four_week_same_weekday_avg(daily_df, pid, latest_date),
                    period_stats=calc_avg_units_sold_in_period(
                        daily_df, pid, period_start, period_end
                    ),
                    single_item_stats=calc_single_item_sales_stats_in_period(
                        daily_df, pid, period_start, period_end
                    ),
                    weekday_units=calc_avg_units_by_weekday_in_period(
                        daily_df, pid, period_start, period_end
                    ),
                    recommended_order=recommended,
                    current_stock=inv_stock,
                    inventory_summary=inventory_summary,
                )
            )

        chart_images = {k: v for k, v in chart_images.items() if v}

        ctx = MtgReportContext(
            app_title=APP_TITLE,
            period_label=period_label,
            period_start=period_start,
            period_end=period_end,
            latest_store_label=business_latest.strftime("%Y/%m/%d"),
            latest_store_sales=day_total_sales,
            all_foods_units=all_foods_units,
            all_foods_rate=all_foods_rate,
            period_customers=period_customers,
            food_breakdown=food_breakdown,
            customer_stats=customer_stats,
            predicted_customers=int(predicted_customers),
            correction_pct=int(correction_pct),
            correction_factor=correction_factor,
            cold_storage_capacity=COLD_STORAGE_CAPACITY,
            food_sections=food_sections,
            chart_images=chart_images,
            memo=report_memo,
        )
        try:
            pdf_bytes = build_mtg_report_pdf(ctx)
            st.session_state["mtg_pdf_bytes"] = pdf_bytes
            st.session_state["mtg_pdf_filename"] = f"MTG_全フード_{period_end:%Y%m%d}.pdf"
            st.success(
                f"PDFを作成しました（{len(food_sections)} フード分）。下のボタンからダウンロードできます。"
            )
        except Exception as exc:
            st.error(f"PDFの作成に失敗しました: {exc}")

    if st.session_state.get("mtg_pdf_bytes"):
        st.download_button(
            label="📄 PDFをダウンロード",
            data=st.session_state["mtg_pdf_bytes"],
            file_name=st.session_state.get("mtg_pdf_filename", "MTG_report.pdf"),
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="mtg_download_pdf",
        )


def plot_inventory_vertical_calendar(
    projection: pd.DataFrame, product_name: str, max_stock: int = 0
) -> go.Figure:
    """縦型: 日付をY軸に予測消費（棒）と予想在庫（線）。"""
    if projection.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{product_name} — 在庫カレンダー", height=200)
        return fig

    labels = projection["label"].tolist()
    colors = projection["status"].map(
        {"out": "#e94560", "low": "#f0ad4e", "over": "#8e44ad", "ok": "#0f3460"}
    ).tolist()

    fig = go.Figure()
    use_col = "effective_use" if "effective_use" in projection.columns else (
        "planned_use" if "planned_use" in projection.columns else "predicted_use"
    )
    use_vals = projection[use_col].astype(int)
    stock_vals = projection["stock_end"].astype(int)
    fig.add_trace(
        go.Bar(
            y=labels,
            x=use_vals,
            orientation="h",
            name="反映消費",
            marker_color=colors,
            text=[f"{int(v)} 個" for v in use_vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>消費: %{x} 個<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=labels,
            x=stock_vals,
            mode="lines+markers",
            name="日末予想在庫",
            line=dict(color="#16a085", width=2.5),
            marker=dict(size=6),
            hovertemplate="%{y}<br>予想在庫: %{x} 個<extra></extra>",
        )
    )
    x_max = max(int(use_vals.max()), int(stock_vals.max()), 1)
    if int(max_stock) > 0:
        x_max = max(x_max, int(max_stock))
    x_min = min(int(stock_vals.min()), 0)
    if int(max_stock) > 0:
        fig.add_vline(
            x=int(max_stock),
            line_dash="dash",
            line_color="#8e44ad",
            line_width=2,
            annotation_text=f"MAX {int(max_stock):,}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=dict(
            text=f"{product_name} — 在庫消費カレンダー<br><sup>上ほど近い日</sup>",
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        xaxis_title="個数",
        xaxis=dict(range=[x_min - max(20, abs(x_min) * 0.05), x_max * 1.25]),
        yaxis=dict(autorange="reversed", title=""),
        template="plotly_white",
        height=max(440, len(projection) * 40),
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.95)",
        ),
        margin=dict(l=12, r=72, t=72, b=88),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )
    return fig


def _projection_row_date(row: pd.Series) -> date:
    day_val = row["date"]
    if isinstance(day_val, pd.Timestamp):
        return day_val.date()
    if isinstance(day_val, date):
        return day_val
    return pd.to_datetime(day_val).date()


PLAN_CONSUMPTION_COL = "✏️ 計画消費（入力）"
ACTUAL_SALES_COL = "✏️ 実売数（入力）"
DELIVERY_PLAN_COL = "📦 納品数（入力）"
EFFECTIVE_USE_COL = "反映消費"


def _parse_optional_int(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    text = str(val).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return max(0, int(float(text)))
    except (TypeError, ValueError):
        return None


def extract_calendar_plan_from_editor(
    edited_df: pd.DataFrame,
    *,
    today: date,
) -> tuple[dict[date, int], dict[date, int], dict[date, int]]:
    """カレンダー表から計画消費・実売数・納品数を抽出する。"""
    planned_use: dict[date, int] = {}
    actual_sales: dict[date, int] = {}
    delivery_plan: dict[date, int] = {}
    for _, row in edited_df.iterrows():
        day_val = pd.to_datetime(str(row["日付"])).date()
        planned_use[day_val] = max(0, int(row[PLAN_CONSUMPTION_COL]))
        delivery_plan[day_val] = max(0, int(row[DELIVERY_PLAN_COL]))
        if day_val <= today:
            parsed = _parse_optional_int(row.get(ACTUAL_SALES_COL))
            if parsed is not None:
                actual_sales[day_val] = parsed
    return planned_use, actual_sales, delivery_plan


def persist_calendar_plan(
    product_id: str,
    start_date: date,
    end_date: date,
    planned_use: dict[date, int],
    actual_sales: dict[date, int],
    delivery_plan: dict[date, int],
    *,
    today: date,
) -> bool:
    """表示中のカレンダー内容をそのまま保存する。"""
    store.mark_calendar_data_dirty()
    ok = clear_consumption_plan(product_id, start_date, end_date)
    if planned_use:
        ok = save_consumption_plan(product_id, planned_use) and ok
    ok = replace_actual_sales_for_period(
        product_id, start_date, end_date, actual_sales, today=today
    ) and ok
    ok = replace_delivery_plan_for_period(product_id, start_date, end_date, delivery_plan) and ok
    store.mark_calendar_data_dirty()
    return ok


def calendar_editor_revision_key(product_id: str) -> str:
    return f"inv_calendar_rev_{product_id}"


def delivery_weekday_checkbox_key(product_id: str, weekday: int) -> str:
    return f"inv_wd_on_{product_id}_{weekday}"


def delivery_weekdays_from_session(product_id: str) -> list[int]:
    """画面上の納品曜日チェック状態を取得。"""
    return [
        wd_idx
        for wd_idx in range(7)
        if st.session_state.get(delivery_weekday_checkbox_key(product_id, wd_idx), False)
    ]


def persist_delivery_weekdays_from_session(product_id: str) -> bool:
    """納品曜日を CSV に保存（チェック状態から）。"""
    return save_delivery_weekdays(product_id, delivery_weekdays_from_session(product_id))


def on_delivery_weekday_toggle(product_id: str) -> None:
    """チェック変更時に納品曜日を即時保存。"""
    weekdays = delivery_weekdays_from_session(product_id)
    ok = save_delivery_weekdays(product_id, weekdays)
    label = weekday_labels_text(weekdays)
    msg = f"納品曜日を保存しました: {label}"
    if not ok and store.is_cloud_enabled():
        msg += "（クラウド送信に失敗しました）"
    st.session_state[f"inv_wd_notice_{product_id}"] = msg


def calendar_editor_widget_key(product_id: str) -> str:
    rev = int(st.session_state.get(calendar_editor_revision_key(product_id), 0))
    return f"inv_calendar_editor_{product_id}_{rev}"


def bump_calendar_editor_revision(product_id: str) -> None:
    """保存・リセット後に data_editor を初期表示へ戻す。"""
    rev_key = calendar_editor_revision_key(product_id)
    st.session_state[rev_key] = int(st.session_state.get(rev_key, 0)) + 1


def format_delivery_display(row: pd.Series) -> str:
    total = int(row.get("delivery", 0))
    if total <= 0:
        return "—"
    return f"+{total}"


def build_inventory_calendar_editor_df(
    projection: pd.DataFrame,
    delivery_weekdays: list[int] | None = None,
    max_stock: int = 0,
    actual_by_day: dict[date, int] | None = None,
    *,
    today: date | None = None,
) -> pd.DataFrame:
    """縦型カレンダー（実売数・計画消費・納品数列を編集）。"""
    if projection.empty:
        return pd.DataFrame(
            columns=[
                "",
                "日付",
                "曜日",
                "予測消費",
                ACTUAL_SALES_COL,
                PLAN_CONSUMPTION_COL,
                DELIVERY_PLAN_COL,
                EFFECTIVE_USE_COL,
                "入荷",
                "収納MAX",
                "日末在庫",
                "状態",
            ]
        )
    cal = projection.copy()
    status_label = {"ok": "🟢", "low": "🟡", "out": "🔴", "over": "🟣"}
    max_val = int(max_stock) if int(max_stock) > 0 else COLD_STORAGE_CAPACITY
    wd_set = set(delivery_weekdays or [])
    actual_map = actual_by_day or {}
    today_val = today or date.today()

    def row_marker(r: pd.Series) -> str:
        if bool(r.get("is_today")):
            return "今日"
        day_val = _projection_row_date(r)
        if day_val < today_val:
            return "過去"
        if day_val.weekday() in wd_set:
            return "納品日"
        return ""

    cal[""] = cal.apply(row_marker, axis=1)
    cal["日付"] = cal["date"].apply(
        lambda d: _projection_row_date(pd.Series({"date": d})).strftime("%Y-%m-%d")
    )
    cal["状態"] = cal.apply(
        lambda r: status_label.get(str(r["status"]), "🟢")
        + (" 実" if bool(r.get("is_actual")) else "")
        + (" ✎" if bool(r.get("is_manual")) and not bool(r.get("is_actual")) else ""),
        axis=1,
    )
    use_col = "planned_use" if "planned_use" in cal.columns else "predicted_use"
    effective_col = "effective_use" if "effective_use" in cal.columns else use_col
    scheduled = cal["delivery_scheduled"] if "delivery_scheduled" in cal.columns else cal["delivery"]
    actual_inputs: list[Any] = []
    for _, row in cal.iterrows():
        day_val = _projection_row_date(row)
        if day_val <= today_val and day_val in actual_map:
            actual_inputs.append(int(actual_map[day_val]))
        else:
            actual_inputs.append(np.nan)
    return pd.DataFrame(
        {
            "": cal[""],
            "日付": cal["日付"],
            "曜日": cal["曜日"],
            "予測消費": cal["predicted_use"].astype(int),
            ACTUAL_SALES_COL: actual_inputs,
            PLAN_CONSUMPTION_COL: cal[use_col].astype(int),
            DELIVERY_PLAN_COL: scheduled.astype(int),
            EFFECTIVE_USE_COL: cal[effective_col].astype(int),
            "入荷": cal.apply(format_delivery_display, axis=1),
            "収納MAX": max_val,
            "日末在庫": cal["stock_end"].apply(
                lambda v: f"{int(v):,} / {max_val:,}"
            ),
            "状態": cal["状態"],
        }
    )


def get_inventory_row_defaults(inv_df: pd.DataFrame, product_id: str) -> tuple[int, int, int]:
    """在庫CSVから商品ごとの現在庫・安全在庫・収納MAXを取得。"""
    inv_row = inv_df[inv_df["product_id"].astype(str) == str(product_id)]
    stock = int(inv_row["current_stock"].iloc[0]) if not inv_row.empty else 0
    safety = int(inv_row["safety_stock"].iloc[0]) if not inv_row.empty else 10
    max_stock = int(inv_row["max_stock"].iloc[0]) if not inv_row.empty else COLD_STORAGE_CAPACITY
    if max_stock <= 0:
        max_stock = COLD_STORAGE_CAPACITY
    return stock, safety, max_stock


def render_all_products_stock_editor(product_list: pd.DataFrame) -> None:
    """全フードの在庫を商品ごとに登録。"""
    st.markdown("##### 全フードの在庫登録")
    st.caption(
        "商品ごとに **現在の在庫**・**安全在庫**・**収納MAX（在庫上限）** を入力して保存します。"
    )

    h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
    with h1:
        st.caption("**商品名**")
    with h2:
        st.caption("**現在の在庫**")
    with h3:
        st.caption("**安全在庫**")
    with h4:
        st.caption("**収納MAX**")

    inv_df = load_inventory_df()
    stock_rows: list[tuple[str, str, int, int, int]] = []
    form_rev = stock_form_revision()

    with st.form(
        f"all_products_stock_form_{form_rev}",
        clear_on_submit=False,
    ):
        for _, prow in product_list.iterrows():
            pid = str(prow["product_id"])
            pname = str(prow["name"])
            d_stock, d_safety, d_max = get_inventory_row_defaults(inv_df, pid)
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                st.markdown(f"**{pname}**")
            with c2:
                stock_val = st.number_input(
                    "在庫",
                    min_value=0,
                    max_value=9999,
                    value=d_stock,
                    step=1,
                    key=f"all_inv_stock_{pid}_{form_rev}",
                    label_visibility="collapsed",
                )
            with c3:
                safety_val = st.number_input(
                    "安全在庫",
                    min_value=0,
                    max_value=500,
                    value=d_safety,
                    step=1,
                    key=f"all_inv_safety_{pid}_{form_rev}",
                    label_visibility="collapsed",
                )
            with c4:
                max_val = st.number_input(
                    "収納MAX",
                    min_value=1,
                    max_value=9999,
                    value=d_max,
                    step=5,
                    key=f"all_inv_max_{pid}_{form_rev}",
                    label_visibility="collapsed",
                )
            stock_rows.append((pid, pname, int(stock_val), int(safety_val), int(max_val)))

        submitted = st.form_submit_button(
            "全フードの在庫を保存",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        cloud_ok = True
        for pid, pname, stock, safety, max_stock in stock_rows:
            ok = save_product_inventory(pid, pname, stock, safety, max_stock)
            cloud_ok = cloud_ok and ok
        bump_stock_form_revision()
        st.session_state["inv_data_rev"] = int(st.session_state.get("inv_data_rev", 0)) + 1
        if store.is_cloud_enabled() and cloud_ok:
            store.set_persist_notice("全フードの在庫を保存し、クラウドにも同期しました。", "success")
        elif store.is_cloud_enabled():
            store.set_persist_notice(
                "在庫はこの端末に保存しましたが、クラウド同期に失敗しました。",
                "warning",
            )
        else:
            store.set_persist_notice("全フードの在庫を保存しました。", "success")
        st.rerun()


@st.fragment
def _render_inventory_calendar_section(
    daily_df: pd.DataFrame,
    product_list: pd.DataFrame,
) -> None:
    """在庫カレンダー部分のみ再描画し、在庫一覧などの再計算を避ける。"""
    _ = st.session_state.get("inv_data_rev", 0)
    st.markdown("##### カレンダー・納品の設定")
    selected_name = st.selectbox(
        "カレンダーを表示するフード商品",
        product_list["name"].tolist(),
        key="inv_product",
    )
    selected = product_list[product_list["name"] == selected_name].iloc[0]
    product_id = str(selected["product_id"])

    inv_df = load_inventory_df()
    current_stock, safety_stock, max_stock = get_inventory_row_defaults(inv_df, product_id)
    st.caption(
        f"**{selected_name}** の登録: **今日の開始在庫 {current_stock:,}** 個"
        f" / 安全 **{safety_stock:,}** / 収納MAX **{max_stock:,}**"
        " — 上の一覧で保存した数値が起点です。"
        " **日末在庫** はその日の消費・入荷後の数です。"
    )

    st.markdown("##### 納品曜日")
    st.caption(
        "納品がある曜日にチェックを入れます。**変更すると自動で保存**され、次回以降も保持されます。"
        " 納品数はカレンダー表に直接入力してください。"
    )

    weekdays_current = get_delivery_weekdays(product_id)
    if notice := st.session_state.pop(f"inv_wd_notice_{product_id}", None):
        st.success(notice)

    wd_cols = st.columns(7)
    for wd_idx, wd_label in enumerate(WEEKDAY_LABELS_JA):
        with wd_cols[wd_idx]:
            st.checkbox(
                wd_label,
                value=(wd_idx in weekdays_current),
                key=delivery_weekday_checkbox_key(product_id, wd_idx),
                on_change=on_delivery_weekday_toggle,
                args=(product_id,),
            )

    if weekdays_current:
        st.caption(f"登録中: **{weekday_labels_text(weekdays_current)}**（カレンダー左列に「納品日」と表示）")
    else:
        st.caption("納品曜日が未設定です。チェックを入れると自動で保存されます。")

    if is_daily_data_empty(daily_df):
        st.warning("販売データがないため予測消費を算出できません。CSVを取り込んでからご利用ください。")
        return

    today_val = date.today()
    start_date, end_date = calendar_date_range(today_val)
    inv_df = load_inventory_df()
    current_stock, safety_stock, max_stock = get_inventory_row_defaults(inv_df, product_id)

    projection = build_inventory_calendar_projection(
        daily_df,
        product_id,
        today=today_val,
    )
    planned_use = load_manual_planned_use(product_id, start_date, end_date)
    actual_sales = load_manual_actual_sales(product_id, start_date, end_date)
    projection = apply_calendar_inputs(
        projection,
        planned_use,
        actual_sales,
        today=today_val,
        opening_stock_today=int(current_stock),
        safety_stock=int(safety_stock),
        max_stock=int(max_stock),
    )

    avg_use = float(projection["effective_use"].mean()) if "effective_use" in projection.columns else (
        float(projection["planned_use"].mean()) if not projection.empty else 0.0
    )
    stockout_in = days_until_stockout(projection, from_date=today_val)
    future_mask = projection["date"].apply(
        lambda d: _projection_row_date(pd.Series({"date": d})) >= today_val
    )
    min_stock = (
        int(projection.loc[future_mask, "stock_end"].min())
        if future_mask.any()
        else 0
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("反映消費の平均", f"{avg_use:,.1f} 個/日")
    with m2:
        st.metric("収納MAX", f"{max_stock:,} 個")
    with m3:
        st.metric(
            f"期末（{end_date.strftime('%m/%d')}）予想在庫",
            f"{int(projection['stock_end'].iloc[-1]):,} 個",
        )
    with m4:
        st.metric("期間内の最低在庫", f"{min_stock:,} 個")
    with m5:
        if stockout_in is not None:
            st.metric("在庫切れ予測", f"{stockout_in} 日後", delta="要発注", delta_color="inverse")
        else:
            st.metric("在庫切れ予測", "なし", delta="期間内は維持")

    delivery_weekdays = get_delivery_weekdays(product_id)
    editor_df = build_inventory_calendar_editor_df(
        projection,
        delivery_weekdays,
        max_stock,
        actual_sales,
        today=today_val,
    )
    editor_key = calendar_editor_widget_key(product_id)

    st.markdown("##### 縦型カレンダー")
    st.caption(
        f"**過去{CALENDAR_PAST_DAYS}日〜未来{CALENDAR_TOTAL_DAYS - CALENDAR_PAST_DAYS - 1}日**"
        f"の **{CALENDAR_TOTAL_DAYS}日間** を表示します。"
        f" **{ACTUAL_SALES_COL}** は今日以前の実際の販売数（手入力）。"
        f" **{PLAN_CONSUMPTION_COL}** は見込み消費。"
        f" **{DELIVERY_PLAN_COL}** は納品数。"
        " 入力後は必ず **「カレンダーを保存して在庫を再計算」** を押してください（保存しないと消えます）。"
        f" **{EFFECTIVE_USE_COL}** は在庫計算に使った消費数です（実売数入力があれば優先）。"
    )

    with st.form(
        f"inv_calendar_form_{product_id}",
        clear_on_submit=False,
    ):
        edited_df = st.data_editor(
            editor_df,
            key=editor_key,
            column_config={
                ACTUAL_SALES_COL: st.column_config.NumberColumn(
                    ACTUAL_SALES_COL,
                    help="今日以前の実際の販売数。未入力の日は計画消費を在庫計算に使います。",
                    min_value=0,
                    max_value=9999,
                    step=1,
                    format="%d",
                ),
                PLAN_CONSUMPTION_COL: st.column_config.NumberColumn(
                    PLAN_CONSUMPTION_COL,
                    help="その日に使う個数の見込み（主に未来の日）",
                    min_value=0,
                    max_value=9999,
                    step=1,
                    format="%d",
                ),
                DELIVERY_PLAN_COL: st.column_config.NumberColumn(
                    DELIVERY_PLAN_COL,
                    help="その日の納品数（隔週など日ごとに変えられます）",
                    min_value=0,
                    max_value=9999,
                    step=1,
                    format="%d",
                ),
            },
            disabled=["", "日付", "曜日", "予測消費", EFFECTIVE_USE_COL, "入荷", "収納MAX", "日末在庫", "状態"],
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
        )

        btn_save, btn_reset, btn_reset_actual, btn_reset_del = st.columns(4)
        with btn_save:
            save_plan = st.form_submit_button(
                "カレンダーを保存して在庫を再計算",
                type="primary",
            )
        with btn_reset:
            reset_plan = st.form_submit_button("計画消費を予測に戻す")
        with btn_reset_actual:
            reset_actual = st.form_submit_button("実売数をクリア")
        with btn_reset_del:
            reset_delivery = st.form_submit_button("納品数をクリア")

    if save_plan:
        persist_delivery_weekdays_from_session(product_id)
        planned_use, actual_sales_input, delivery_plan = extract_calendar_plan_from_editor(
            edited_df, today=today_val
        )
        cloud_ok = persist_calendar_plan(
            product_id,
            start_date,
            end_date,
            planned_use,
            actual_sales_input,
            delivery_plan,
            today=today_val,
        )
        bump_calendar_editor_revision(product_id)
        if store.is_cloud_enabled() and cloud_ok:
            store.set_persist_notice("カレンダーを保存し、クラウドにも同期しました。", "success")
        elif store.is_cloud_enabled():
            store.set_persist_notice(
                "カレンダーはこの端末に保存しましたが、クラウド同期に失敗しました。",
                "warning",
            )
        else:
            store.set_persist_notice("カレンダーを保存し、在庫見込みを更新しました。", "success")
        st.rerun()

    if reset_plan:
        clear_consumption_plan(product_id, start_date, end_date)
        bump_calendar_editor_revision(product_id)
        st.success("計画消費を予測に戻しました。")
        st.rerun()

    if reset_actual:
        past_end = min(end_date, today_val)
        clear_actual_sales(product_id, start_date, past_end)
        bump_calendar_editor_revision(product_id)
        st.success("表示期間の実売数（手入力）をクリアしました。")
        st.rerun()

    if reset_delivery:
        replace_delivery_plan_for_period(product_id, start_date, end_date, {})
        bump_calendar_editor_revision(product_id)
        st.success("表示期間の納品数をクリアしました。")
        st.rerun()

    st.caption(
        "🔴 不足 · 🟡 注意 · 🟣 収納超過 · 実 実売数反映 · ✎ 計画手入力 · 日末在庫は「現在/MAX」表示"
    )

    st.markdown("##### 消費と予想在庫")
    st.plotly_chart(
        plot_inventory_vertical_calendar(projection, selected_name, max_stock),
        use_container_width=True,
        config=_PLOTLY_CHART_CONFIG,
    )


def render_inventory_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">在庫管理</p>', unsafe_allow_html=True)
    st.info(
        "**在庫** →「全フードの在庫を保存」、**カレンダー** →「カレンダーを保存して在庫を再計算」"
        " でそれぞれ保存してください。"
        " サイドバーの **「すべてのデータを保存」** でクラウドにも送れます。"
    )
    st.caption(
        "対象は **販売中のフード商品** です（サイドバー「商品・紐づけ設定」で追加した新商品も自動で表示されます）。"
        " **納品曜日** で曜日を登録し、**納品数** は下のカレンダー表に直接入力します。"
        " **計画消費** も同じ表で編集でき、保存後に日末在庫が再計算されます。"
    )

    init_inventory_csv(products_df)
    sync_inventory_products(products_df)
    init_deliveries_csv()
    init_consumption_plan_csv()
    init_delivery_plan_csv()
    init_actual_sales_csv()
    product_list = active_products(products_df)
    render_all_products_stock_editor(product_list)
    _render_inventory_calendar_section(daily_df, product_list)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="wide")
    inject_css()

    if store.is_cloud_enabled():
        first_sync = not st.session_state.get("_cloud_initial_sync")
        if first_sync:
            if store.is_local_data_empty():
                store.ensure_cloud_sync(force=True, force_pull=True)
            else:
                store.push_all_to_cloud()
            st.session_state["_cloud_initial_sync"] = True
            clear_data_caches()
        else:
            synced = store.ensure_cloud_sync()
            if synced > 0:
                clear_data_caches()

    if not st.session_state.get("_data_files_initialized"):
        ensure_data_files()
        st.session_state["_data_files_initialized"] = True
        if store.is_cloud_enabled():
            store.push_all_to_cloud()

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

    if notice := store.pop_persist_notice():
        level = notice.get("level", "info")
        message = notice.get("message", "")
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)

    with st.sidebar:
        st.markdown("### データ管理")
        st.caption(f"登録日数: **{len(list_recorded_dates(daily_df))}** 日")
        if st.button("ゴミ日データを削除", type="secondary", use_container_width=True):
            n = purge_meaningless_days()
            if n:
                clear_data_caches()
                st.success(f"販売ゼロ・極小の日を {n} 日分削除しました。")
            else:
                st.info("削除対象の日はありませんでした。")
            st.rerun()
        if st.button("全データをリセット", type="secondary", use_container_width=True):
            reset_application_data()
            st.session_state["_data_files_initialized"] = True
            _clear_pending_csv_payload()
            clear_data_caches()
            if store.is_cloud_enabled():
                store.push_all_to_cloud()
            st.success("データを初期化しました。")
            st.rerun()

        st.divider()
        st.markdown("### データ同期")
        st.caption(store.cloud_status_label())
        if store.is_cloud_enabled():
            st.caption(
                "データは端末に保存されたまま維持されます。"
                " 通常の同期では古いクラウドデータで上書きしません。"
            )
            ok_conn, conn_err = store.probe_cloud_connection()
            if not ok_conn:
                st.error(f"クラウド接続エラー: {conn_err}")
            if st.button("すべてのデータを保存", type="primary", use_container_width=True):
                uploaded, upload_errors = store.push_all_to_cloud()
                if upload_errors:
                    st.warning("一部のファイルを送信できませんでした。")
                    for msg in upload_errors[:3]:
                        st.caption(msg)
                else:
                    store.set_persist_notice(
                        f"すべてのデータ（{uploaded} 件）をクラウドに保存しました。",
                        "success",
                    )
                    st.rerun()
            if st.button("クラウドから最新を取得", type="secondary", use_container_width=True):
                store.clear_session_dirty()
                store.ensure_cloud_sync(force=True, force_pull=True)
                clear_data_caches()
                st.rerun()
            if st.button("この端末のデータをクラウドへ送信", type="secondary", use_container_width=True):
                uploaded, upload_errors = store.push_all_to_cloud()
                if upload_errors:
                    st.warning("一部のファイルを送信できませんでした。")
                    for msg in upload_errors[:3]:
                        st.caption(msg)
                else:
                    st.success(f"クラウドへ {uploaded} 件のデータを送信しました。")
        else:
            setup = store.get_cloud_setup_status()
            if store.is_ephemeral_host():
                st.warning("クラウド未接続 — データが消える原因はここです")
            st.caption(
                f"診断: supabaseセクション={'あり' if setup['has_section'] else 'なし'}"
                f" / url={'OK' if setup['has_url'] else 'なし'}"
                f" / key={'OK' if setup['has_key'] else 'なし'}"
                + (f"（{setup['key_length']}文字）" if setup.get("key_length") else "")
            )
            if setup.get("error_hint"):
                st.caption(str(setup["error_hint"]))
            if setup["has_section"] and not setup["has_key"]:
                st.error("Secrets に [supabase] はありますが **key** が読めません。")
            elif setup["has_section"] and not setup["has_url"]:
                st.error("Secrets に [supabase] はありますが **url** が読めません。")
            elif not setup["has_section"]:
                st.error("Secrets に **[supabase]** セクションがまだありません。")
            with st.expander(
                "Supabase 設定手順（Streamlit Cloud）",
                expanded=store.is_ephemeral_host(),
            ):
                st.markdown(
                    """
1. ブラウザで **[share.streamlit.io](https://share.streamlit.io)** を開く  
2. このアプリのカード右上 **︙** → **Settings**  
3. 左メニュー **Secrets** をクリック  
4. 下の3行を**そのまま**貼り付け（`url` / `key` は自分の値に）  

```toml
[supabase]
url = "https://xxxx.supabase.co"
key = "sb_secret_..."
bucket = "order-app-data"
```

5. **Save** を押す  
6. 右上 **︙** → **Reboot app**（再起動）  
7. このページを **再読み込み（F5）**

**成功すると** 上に「ローカル優先・クラウド同期 ON」と  
「すべてのデータを保存」ボタンが出ます。

**Supabase 側:** Storage に `order-app-data` バケットを作成してください。  
**キー:** Project Settings → API の **Secret key**（`sb_secret_...`）  
または **Legacy service_role**（`eyJ...` で始まる JWT）でも可

**注意:** GitHub の Secrets ではなく、**Streamlit Cloud の Secrets** に貼ってください。
                    """
                )
            if not store.is_ephemeral_host():
                st.caption(
                    "ローカル実行時は `.streamlit/secrets.toml` に同じ内容を書いてください。"
                )

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
        main_tab = st.radio(
            "メイン画面",
            MAIN_TAB_OPTIONS,
            horizontal=True,
            label_visibility="collapsed",
            key="main_work_tab",
        )
        if main_tab == MAIN_TAB_OPTIONS[0]:
            render_square_upload_tab(products_df, daily_df)
        elif main_tab == MAIN_TAB_OPTIONS[1]:
            render_dashboard_tab(products_df, daily_df)
        elif main_tab == MAIN_TAB_OPTIONS[2]:
            render_inventory_tab(products_df, daily_df)
        else:
            render_mtg_report_tab(products_df, daily_df)


if __name__ == "__main__":
    main()
