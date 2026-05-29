"""
Square CSV連携 + 手動修正型 フード発注管理アプリ
- Square売上CSVアップロードで一括取り込み
- 過去データの確認・修正 / 商品管理
- ダッシュボード・発注予測
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
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
APP_TITLE = "Square CSV連携 フード発注管理"
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

EMPTY_DATA_MESSAGE = "データがありません。SquareのCSVファイルをアップロードしてください。"

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

# Square CSV 列名の候補（日本語・英語）— 優先度順
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": [
        "date",
        "日付",
        "時期",
        "取引日",
        "取引日時",
        "営業日",
        "day",
        "売上日",
        "datetime",
        "date time",
    ],
    "time": ["time", "時刻", "取引時刻", "時間"],
    "item": [
        "item",
        "item name",
        "商品名",
        "アイテム名",
        "アイテム",
        "品目",
        "品名",
        "メニュー",
        "商品",
        "メニュー名",
        "items",
    ],
    "quantity": [
        "qty",
        "quantity",
        "数量",
        "販売数",
        "個数",
        "sold",
        "販売個数",
        "販売数量",
        "items sold",
        "売上数量",
        "点数",
    ],
    "net_sales": ["net sales", "ネット売上", "純売上", "net", "正味売上", "売上（税込）"],
    "gross_sales": ["gross sales", "総売上", "売上", "gross", "売上高", "総売上高"],
    "customers": [
        "customers",
        "客数",
        "来店客数",
        "取引数",
        "transactions",
        "transaction count",
        "総客数",
        "取引件数",
    ],
    "total_sales": ["total sales", "店舗売上", "総売上高", "売上合計", "店舗総売上", "合計売上"],
}


@dataclass
class DayImport:
    day: date
    total_sales: int
    total_customers: int
    units_by_product: dict[str, int]


@dataclass
class DaySummary:
    total_sales: int
    total_customers: int


SUMMARY_CUSTOMER_ROW_KEYWORDS = ["取引", "transaction", "客数", "注文件数", "件数"]
SUMMARY_SALES_ROW_KEYWORDS = ["売上", "sales", "gross", "net", "収入", "合計金額"]

# Square「商品売上サマリー」の商品名 → アプリ登録名（長いパターンを先に評価）
PRODUCT_NAME_ALIASES: dict[str, list[str]] = {
    "パフェ": ["ワッフル パフェ", "ワッフル　パフェ"],
    "ワッフル": ["リエージュワッフル"],
    "抹茶チーズケーキ": ["抹茶チーズケーキ"],
    "チーズケーキ": ["トンカチーズケーキ"],
    "バナナパウンドケーキ": ["バナナパウンドケーキ"],
    "レモンパウンドケーキ": ["レモンポピーシード パウンドケーキ", "レモンパウンドケーキ"],
    "季節のソースとグラノーラ": [
        "オリジナルグラノーラ＋ヨーグルトと季節のジャム",
        "オリジナルグラノーラ+ヨーグルトと季節のジャム",
    ],
    "グラノーラ": ["オリジナルグラノーラ＋ミルク", "オリジナルグラノーラ+ミルク"],
    "オーバーナイトグラノーラ": ["オーバーナイト オーツ", "オーバーナイトオーツ"],
}

KV_CUSTOMER_METRIC_LABELS = ["売上取引履歴", "総売上取引履歴", "受取合計額の取引履歴"]
KV_SALES_METRIC_LABELS = ["純売上高", "総売上高", "売上合計", "受取合計額", "正味売上"]
KV_SUMMARY_LABEL_HINTS = KV_CUSTOMER_METRIC_LABELS + KV_SALES_METRIC_LABELS + [
    "総売上",
    "純売上",
    "取引履歴",
    "ギフトカード",
    "税金",
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
            <strong>📂 {EMPTY_DATA_MESSAGE}</strong><br>
            「Square CSVアップロード」タブから、Squareダッシュボードでエクスポートした売上CSVを取り込んでください。
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_empty_daily_csv() -> None:
    """営業データCSVをヘッダーのみの空状態にする。"""
    pd.DataFrame(columns=DAILY_SALES_COLUMNS).to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")


def clear_all_daily_sales() -> None:
    """デモ・過去の営業データをすべて削除する。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_empty_daily_csv()


def normalize_col(name: str) -> str:
    s = str(name).replace("\ufeff", "").strip().lower()
    return re.sub(r"\s+", " ", s)


def find_column(columns: list[str], keys: list[str]) -> str | None:
    """列名をあいまいマッチング（完全一致を最優先）。"""
    candidates: list[tuple[int, str]] = []
    for original in columns:
        col_norm = normalize_col(original)
        if not col_norm or col_norm == "nan":
            continue
        for priority, key in enumerate(keys):
            score = 0
            if col_norm == key:
                score = 1000 - priority
            elif col_norm.startswith(key) or key.startswith(col_norm):
                score = 800 - priority
            elif key in col_norm:
                score = 600 - priority
            elif col_norm in key:
                score = 500 - priority
            if score > 0:
                candidates.append((score, original))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def detect_column_mapping(columns: list[str]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {
        key: find_column(columns, aliases) for key, aliases in COLUMN_ALIASES.items()
    }
    mapping["matrix_item_column"] = find_matrix_item_column(columns)
    date_cols = get_matrix_date_columns(columns)
    mapping["matrix_date_columns"] = ", ".join(date_cols[:8]) + (" ..." if len(date_cols) > 8 else "") if date_cols else None
    mapping["format"] = "マトリックス（日付=横）" if (find_matrix_item_column(columns) and date_cols) else "縦持ち"
    return mapping


MATRIX_ITEM_ALIASES = COLUMN_ALIASES["item"] + ["カテゴリ", "category", "種別", "メニュー区分", "分類"]
SKIP_ROW_KEYWORDS = ["合計", "総計", "小計", "計", "total", "subtotal", "カテゴリ合計"]


def parse_column_as_date(col_name: str) -> date | None:
    """列名が日付（例: 2026/5/1）かどうかを判定する。"""
    s = str(col_name).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    if find_column([s], MATRIX_ITEM_ALIASES + COLUMN_ALIASES["quantity"]):
        return None
    parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    if pd.isna(parsed):
        return None
    if parsed.year < 2000 or parsed.year > 2100:
        return None
    return parsed.date()


def get_matrix_date_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if parse_column_as_date(c) is not None]


def find_matrix_item_column(columns: list[str]) -> str | None:
    return find_column(columns, MATRIX_ITEM_ALIASES)


def is_matrix_format_df(df: pd.DataFrame) -> bool:
    """日付が横方向・商品名が縦方向のマトリックス形式か。"""
    cols = [str(c) for c in df.columns]
    item_col = find_matrix_item_column(cols)
    date_cols = get_matrix_date_columns(cols)
    return item_col is not None and len(date_cols) >= 1


def is_probable_matrix_header_row(row_vals: list[str]) -> bool:
    row_cols = [str(v).strip() for v in row_vals if str(v).strip() and str(v).lower() != "nan"]
    has_item = find_column(row_cols, MATRIX_ITEM_ALIASES) is not None
    date_count = len(get_matrix_date_columns(row_cols))
    return has_item and date_count >= 2


def is_probable_header_row(row_vals: list[str]) -> bool:
    """縦持ち形式、またはマトリックス形式のヘッダー行か。"""
    if is_probable_matrix_header_row(row_vals):
        return True
    row_cols = [str(v).strip() for v in row_vals if str(v).strip() and str(v).lower() != "nan"]
    if len(row_cols) < 2:
        return False
    has_date = find_column(row_cols, COLUMN_ALIASES["date"]) is not None
    has_item = find_column(row_cols, COLUMN_ALIASES["item"]) is not None
    has_qty = find_column(row_cols, COLUMN_ALIASES["quantity"]) is not None
    return has_date and (has_item or has_qty)


def is_summary_row_label(name: str) -> bool:
    label = str(name).strip().lower()
    return any(k in label for k in SKIP_ROW_KEYWORDS)


def decode_uploaded_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def detect_csv_separator(text: str) -> str:
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.count("\t") >= line.count(","):
            return "\t"
        return ","
    return ","


def parse_yen_to_int(val: Any) -> int:
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-", "—"):
        return 0
    negative = "(" in s or s.startswith("-")
    s = re.sub(r"[￥¥,\s\"']", "", s)
    s = s.replace("(", "").replace(")", "")
    if not s:
        return 0
    try:
        num = int(float(s))
    except ValueError:
        return 0
    return -num if negative else num


def parse_count_metric(val: Any) -> int:
    s = str(val).strip().replace(",", "").replace('"', "")
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else 0


def cell_value_is_yen(val: Any) -> bool:
    s = str(val).strip()
    return "￥" in s or "¥" in s or ("(" in s and ")" in s and re.search(r"\d", s))


def parse_matrix_cell(val: Any) -> tuple[int, bool]:
    """(数値, 金額表記か) — 金額の場合は円単位の整数。"""
    if cell_value_is_yen(val):
        return parse_yen_to_int(val), True
    num = int(pd.to_numeric(val, errors="coerce") or 0)
    return num, False


def normalize_square_kv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Square sales-summary を metric / value の2列に正規化。"""
    if df.empty:
        return df
    if "metric" in df.columns and "value" in df.columns:
        return df
    if df.shape[1] < 2:
        return df
    return pd.DataFrame(
        {
            "metric": df.iloc[:, 0].astype(str).str.strip(),
            "value": df.iloc[:, 1],
        }
    )


def read_square_sales_summary_csv(uploaded_file: Any) -> pd.DataFrame:
    """Square sales-summary（先頭メタ行付き・2列CSV）を読み込む。"""
    text = decode_uploaded_bytes(uploaded_file.getvalue())
    df = pd.read_csv(io.StringIO(text))
    return normalize_square_kv_dataframe(df)


def kv_metric_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "metric" in df.columns:
        return df["metric"].astype(str).str.strip(), df["value"]
    return df.iloc[:, 0].astype(str).str.strip(), df.iloc[:, 1]


def is_square_kv_summary_df(df: pd.DataFrame) -> bool:
    """Square sales-summary（指標名×値の2列・期間集計）。"""
    if df.empty or df.shape[1] < 2:
        return False
    cols = [str(c) for c in df.columns]
    if get_matrix_date_columns(cols):
        return False
    labels, _values = kv_metric_series(df)
    hits = sum(1 for lab in labels if any(h in lab for h in KV_SUMMARY_LABEL_HINTS))
    return hits >= 2


def extract_period_summary_from_kv(df: pd.DataFrame) -> tuple[int, int]:
    """期間合計の (総客数=取引件数, 店舗総売上)。"""
    df = normalize_square_kv_dataframe(df)
    labels, values = kv_metric_series(df)
    metric_map = dict(zip(labels, values))

    customers = 0
    for key in KV_CUSTOMER_METRIC_LABELS:
        if key in metric_map:
            customers = parse_count_metric(metric_map[key])
            break

    sales = 0
    for key in KV_SALES_METRIC_LABELS:
        if key in metric_map:
            sales = parse_yen_to_int(metric_map[key])
            break

    if customers == 0 and sales == 0:
        raise ValueError("売上サマリーCSVから期間合計（取引件数・売上）を抽出できませんでした。")
    return customers, sales


def distribute_period_summary_to_days(
    period_customers: int,
    period_sales: int,
    daily_weights: dict[date, float],
) -> dict[date, DaySummary]:
    if not daily_weights:
        raise ValueError("商品別CSVの日付が無いため、期間サマリーを日別に配分できません。")
    total_w = sum(daily_weights.values()) or 1.0
    by_day: dict[date, DaySummary] = {}
    allocated_sales = 0
    allocated_customers = 0
    days_sorted = sorted(daily_weights.keys())
    for i, day_val in enumerate(days_sorted):
        w = daily_weights[day_val]
        if i == len(days_sorted) - 1:
            day_sales = period_sales - allocated_sales
            day_customers = period_customers - allocated_customers
        else:
            ratio = w / total_w
            day_sales = int(period_sales * ratio)
            day_customers = int(period_customers * ratio)
            allocated_sales += day_sales
            allocated_customers += day_customers
        by_day[day_val] = DaySummary(total_sales=max(0, day_sales), total_customers=max(0, day_customers))
    return by_day


def matrix_to_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """マトリックス形式 → 縦持ち（日付, 商品名, 数量）へ変換。"""
    cols = [str(c) for c in df.columns]
    item_col = find_matrix_item_column(cols)
    date_cols = get_matrix_date_columns(cols)
    if not item_col or not date_cols:
        raise ValueError("マトリックス形式の列構成を認識できませんでした。")

    long_df = df[[item_col] + date_cols].copy()
    long_df = long_df.melt(id_vars=[item_col], value_vars=date_cols, var_name="_date_col", value_name="quantity")
    long_df["date"] = long_df["_date_col"].apply(lambda x: parse_column_as_date(str(x)))
    long_df = long_df.dropna(subset=["date"])
    long_df["product_name"] = long_df[item_col].astype(str).str.strip()
    parsed = long_df["quantity"].apply(parse_matrix_cell)
    long_df["quantity"] = parsed.apply(lambda x: x[0])
    long_df["is_revenue"] = parsed.apply(lambda x: x[1])
    long_df = long_df[long_df["product_name"].astype(str).str.len() > 0]
    long_df = long_df[~long_df["product_name"].apply(is_summary_row_label)]
    return long_df[["date", "product_name", "quantity", "is_revenue"]]


def is_customer_metric_label(label: str) -> bool:
    text = str(label).strip().lower()
    if not text or is_summary_row_label(label):
        return False
    if any(k in text for k in SUMMARY_SALES_ROW_KEYWORDS) and "取引" not in text:
        return False
    return any(k in text for k in SUMMARY_CUSTOMER_ROW_KEYWORDS)


def is_sales_metric_label(label: str) -> bool:
    text = str(label).strip().lower()
    if not text or is_summary_row_label(label):
        return False
    if is_customer_metric_label(label):
        return False
    return any(k in text for k in SUMMARY_SALES_ROW_KEYWORDS)


def parse_matrix_units_by_day(
    df: pd.DataFrame, products_df: pd.DataFrame
) -> tuple[dict[date, dict[str, int]], dict[date, float], list[str]]:
    """商品別マトリックスCSVから日別・商品別販売数と日別売上ウェイトを抽出。"""
    warnings: list[str] = ["商品別マトリックスCSV（Square商品売上サマリー）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    if not is_matrix_format_df(prepared):
        raise ValueError("商品別CSVはマトリックス形式（日付が横方向）である必要があります。")

    long_df = matrix_to_long_dataframe(prepared)
    if long_df.empty:
        raise ValueError("商品別マトリックスCSVから販売データを抽出できませんでした。")

    product_names = products_df["name"].astype(str).tolist()
    price_map = dict(zip(products_df["name"], products_df["unit_price"]))
    units_by_day: dict[date, dict[str, int]] = {}
    revenue_by_day: dict[date, float] = {}
    used_yen = False

    for day_val, day_group in long_df.groupby("date"):
        units = {name: 0 for name in product_names}
        day_revenue = 0.0
        unmapped: list[str] = []
        for _, row in day_group.iterrows():
            matched = match_product_name(row["product_name"], product_names)
            if not matched:
                unmapped.append(str(row["product_name"]))
                continue
            raw = int(row["quantity"])
            unit_price = int(price_map.get(matched, 0) or 0)
            if bool(row.get("is_revenue", False)):
                used_yen = True
                day_revenue += raw
                add_units = max(0, round(raw / unit_price)) if unit_price > 0 else 0
            else:
                add_units = raw
                day_revenue += raw * unit_price
            units[matched] = units.get(matched, 0) + add_units
        if unmapped:
            sample = ", ".join(sorted(set(unmapped))[:5])
            warnings.append(f"{day_val}: 未登録商品をスキップ（例: {sample}）")
        units_by_day[day_val] = units
        revenue_by_day[day_val] = max(day_revenue, 1.0) if sum(units.values()) > 0 else 0.0

    if used_yen:
        warnings.append("セルが金額（¥）表記のため、登録単価から販売数を換算しました。")

    return units_by_day, revenue_by_day, warnings


def parse_sales_summary_vertical(df: pd.DataFrame) -> tuple[dict[date, DaySummary], list[str]]:
    """売上サマリー（縦持ち・日付列あり）から日別の総売上・総客数を抽出。"""
    warnings: list[str] = ["売上サマリーCSV（縦持ち）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    cols = [str(c) for c in prepared.columns.tolist()]

    date_col = find_column(cols, COLUMN_ALIASES["date"])
    if not date_col:
        raise ValueError("売上サマリーCSVに日付列が見つかりません。")

    customers_col = find_column(cols, COLUMN_ALIASES["customers"])
    total_sales_col = find_column(cols, COLUMN_ALIASES["total_sales"])
    net_sales_col = find_column(cols, COLUMN_ALIASES["net_sales"])
    gross_sales_col = find_column(cols, COLUMN_ALIASES["gross_sales"])
    sales_col = total_sales_col or gross_sales_col or net_sales_col

    work = prepared.copy()
    work["_parsed_date"] = build_datetime_series(work, date_col, find_column(cols, COLUMN_ALIASES["time"]))
    work = work.dropna(subset=["_parsed_date"])
    work["_day"] = work["_parsed_date"].dt.date

    by_day: dict[date, DaySummary] = {}
    for day_val, group in work.groupby("_day"):
        customers = 0
        if customers_col:
            customers = int(pd.to_numeric(group[customers_col], errors="coerce").fillna(0).max())
        sales = 0
        if sales_col:
            sales = int(pd.to_numeric(group[sales_col], errors="coerce").fillna(0).max())
        if customers == 0 and sales == 0:
            continue
        by_day[day_val] = DaySummary(total_sales=sales, total_customers=customers)

    if not by_day:
        raise ValueError("売上サマリーCSVから有効な日次データを抽出できませんでした。")
    return by_day, warnings


def parse_sales_summary_matrix(df: pd.DataFrame) -> tuple[dict[date, DaySummary], list[str]]:
    """売上サマリー（指標が行・日付が列のマトリックス）から抽出。"""
    warnings: list[str] = ["売上サマリーCSV（マトリックス形式）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    if not is_matrix_format_df(prepared):
        raise ValueError("売上サマリーCSVの形式を認識できませんでした。")

    item_col = find_matrix_item_column([str(c) for c in prepared.columns])
    date_cols = get_matrix_date_columns([str(c) for c in prepared.columns])
    if not item_col or not date_cols:
        raise ValueError("売上サマリーCSVの列構成を認識できませんでした。")

    by_day: dict[date, DaySummary] = {dcol_date: DaySummary(0, 0) for dcol in date_cols if (dcol_date := parse_column_as_date(dcol))}

    for _, row in prepared.iterrows():
        label = str(row[item_col]).strip()
        if not label or is_summary_row_label(label):
            continue
        for dcol in date_cols:
            day_val = parse_column_as_date(dcol)
            if not day_val:
                continue
            val = int(pd.to_numeric(row[dcol], errors="coerce") or 0)
            current = by_day.setdefault(day_val, DaySummary(0, 0))
            if is_customer_metric_label(label):
                by_day[day_val] = DaySummary(current.total_sales, max(current.total_customers, val))
            elif is_sales_metric_label(label):
                by_day[day_val] = DaySummary(max(current.total_sales, val), current.total_customers)

    by_day = {d: s for d, s in by_day.items() if s.total_sales > 0 or s.total_customers > 0}
    if not by_day:
        raise ValueError("売上サマリーCSVから総売上・取引件数を抽出できませんでした。")
    return by_day, warnings


def parse_sales_summary_kv(
    df: pd.DataFrame, daily_weights: dict[date, float]
) -> tuple[dict[date, DaySummary], list[str]]:
    """Square sales-summary（期間集計・2列）を商品別日次ウェイトで日別配分。"""
    warnings: list[str] = [
        "売上サマリーCSV（Square期間集計形式）を読み込み、商品別売上の日次比率で客数・総売上を配分しました。",
    ]
    period_customers, period_sales = extract_period_summary_from_kv(df)
    by_day = distribute_period_summary_to_days(period_customers, period_sales, daily_weights)
    return by_day, warnings


def parse_sales_summary_csv(
    df: pd.DataFrame,
    daily_weights: dict[date, float] | None = None,
) -> tuple[dict[date, DaySummary], list[str]]:
    df = normalize_square_kv_dataframe(df)
    if is_square_kv_summary_df(df):
        if not daily_weights:
            raise ValueError(
                "売上サマリーは期間合計形式です。商品売上サマリーCSVと一緒にアップロードしてください。"
            )
        return parse_sales_summary_kv(df, daily_weights)

    prepared = prepare_square_dataframe(df)
    if is_matrix_format_df(prepared):
        item_col = find_matrix_item_column([str(c) for c in prepared.columns])
        if item_col:
            has_metric_row = any(
                is_customer_metric_label(str(row[item_col])) or is_sales_metric_label(str(row[item_col]))
                for _, row in prepared.iterrows()
            )
            if has_metric_row:
                return parse_sales_summary_matrix(prepared)
    return parse_sales_summary_vertical(prepared)


def classify_uploaded_csv(
    df: pd.DataFrame, products_df: pd.DataFrame, filename: str = ""
) -> str:
    """product_matrix / sales_summary / unknown"""
    name_lower = (filename or "").lower()
    if "sales-summary" in name_lower or ("売上サマリー" in name_lower and "商品" not in name_lower):
        return "sales_summary"

    if is_square_kv_summary_df(df):
        return "sales_summary"

    prepared = prepare_square_dataframe(df)
    product_names = products_df["name"].astype(str).tolist()

    if "商品売上" in name_lower and is_matrix_format_df(prepared):
        return "product_matrix"

    if is_matrix_format_df(prepared):
        item_col = find_matrix_item_column([str(c) for c in prepared.columns])
        if not item_col:
            return "unknown"
        product_rows = 0
        metric_rows = 0
        for _, row in prepared.iterrows():
            label = str(row[item_col]).strip()
            if not label or is_summary_row_label(label):
                continue
            if match_product_name(label, product_names):
                product_rows += 1
            elif is_customer_metric_label(label) or is_sales_metric_label(label):
                metric_rows += 1
        if product_rows > 0 and product_rows >= metric_rows:
            return "product_matrix"
        if metric_rows > 0:
            return "sales_summary"
        matrix_cols = [str(c) for c in prepared.columns]
        if "商品売上" in name_lower or find_column(matrix_cols, ["商品名"]):
            return "product_matrix"
        return "product_matrix"

    cols = [str(c) for c in prepared.columns]
    date_col = find_column(cols, COLUMN_ALIASES["date"])
    sales_col = (
        find_column(cols, COLUMN_ALIASES["total_sales"])
        or find_column(cols, COLUMN_ALIASES["gross_sales"])
        or find_column(cols, COLUMN_ALIASES["net_sales"])
    )
    customers_col = find_column(cols, COLUMN_ALIASES["customers"])
    if date_col and (sales_col or customers_col):
        return "sales_summary"
    if date_col and find_column(cols, COLUMN_ALIASES["item"]) and find_column(cols, COLUMN_ALIASES["quantity"]):
        return "product_matrix"
    return "unknown"


def merge_dual_csv_imports(
    units_by_day: dict[date, dict[str, int]],
    summary_by_day: dict[date, DaySummary],
    products_df: pd.DataFrame,
) -> tuple[list[DayImport], list[str]]:
    """商品別データと売上サマリーを日付キーで結合。"""
    warnings: list[str] = ["ルートB: 2枚のCSVを日付で結合しました。"]
    product_names = products_df["name"].astype(str).tolist()
    price_map = dict(zip(products_df["name"], products_df["unit_price"]))
    all_days = sorted(set(units_by_day) | set(summary_by_day))

    if not all_days:
        raise ValueError("結合できる日付データがありません。")

    only_products = set(units_by_day) - set(summary_by_day)
    only_summary = set(summary_by_day) - set(units_by_day)
    if only_products:
        warnings.append(f"売上サマリーに無い日付: {len(only_products)}日（販売数のみ反映・客数/売上は推定）。")
    if only_summary:
        warnings.append(f"商品別CSVに無い日付: {len(only_summary)}日（客数/売上のみ反映）。")

    imports: list[DayImport] = []
    for day_val in all_days:
        units = units_by_day.get(day_val, {name: 0 for name in product_names})
        summary = summary_by_day.get(day_val)

        if summary:
            total_sales = int(summary.total_sales)
            total_customers = int(summary.total_customers)
        else:
            total_sales = int(sum(units.get(n, 0) * int(price_map.get(n, 0)) for n in product_names))
            total_customers = max(100, int(sum(units.values()) / 0.12)) if sum(units.values()) > 0 else 0

        if not summary or total_sales == 0:
            est_sales = int(sum(units.get(n, 0) * int(price_map.get(n, 0)) for n in product_names))
            if total_sales == 0 and est_sales > 0:
                total_sales = est_sales
        if not summary or total_customers == 0:
            if sum(units.values()) > 0:
                total_customers = max(total_customers, int(sum(units.values()) / 0.12))

        imports.append(DayImport(day_val, total_sales, total_customers, units))

    return imports, warnings


def load_uploaded_dataframe(uploaded_file: Any, products_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """ファイル名と内容から種別を判定し、適切に読み込む。"""
    name = getattr(uploaded_file, "name", "") or ""
    name_lower = name.lower()
    raw = uploaded_file.getvalue()

    if "sales-summary" in name_lower or ("売上サマリー" in name and "商品" not in name):
        df = read_square_sales_summary_csv(uploaded_file)
        return classify_uploaded_csv(df, products_df, name), df

    df = read_uploaded_csv(uploaded_file)
    return classify_uploaded_csv(df, products_df, name), df


def parse_dual_csv_upload(
    uploaded_files: list[Any], products_df: pd.DataFrame
) -> tuple[list[DayImport], list[str]]:
    """最大2枚: 商品別マトリックス + 売上サマリーを結合。"""
    if len(uploaded_files) > 2:
        raise ValueError("アップロードは2ファイルまでです。")

    classified: list[tuple[str, pd.DataFrame, str]] = []
    for f in uploaded_files:
        kind, df = load_uploaded_dataframe(f, products_df)
        classified.append((f.name, df, kind))

    if len(classified) == 1:
        return parse_square_csv(classified[0][1], products_df)

    product_file = next((x for x in classified if x[2] == "product_matrix"), None)
    summary_file = next((x for x in classified if x[2] == "sales_summary"), None)

    if not product_file or not summary_file:
        kinds = ", ".join(f"{x[0]}→{x[2]}" for x in classified)
        raise ValueError(
            "2枚の内訳を認識できませんでした。商品売上サマリーCSVと売上サマリーCSVをアップロードしてください。"
            f"（判定: {kinds}）"
        )

    units_by_day, revenue_by_day, w1 = parse_matrix_units_by_day(product_file[1], products_df)
    summary_by_day, w2 = parse_sales_summary_csv(summary_file[1], daily_weights=revenue_by_day)
    imports, w3 = merge_dual_csv_imports(units_by_day, summary_by_day, products_df)
    return imports, w1 + w2 + w3


def parse_matrix_format(df: pd.DataFrame, products_df: pd.DataFrame) -> tuple[list[DayImport], list[str]]:
    """横持ちマトリックス（日付=列、商品=行）を日次データへ変換（単体アップロード用）。"""
    units_by_day, _revenue_by_day, warnings = parse_matrix_units_by_day(df, products_df)
    price_map = dict(zip(products_df["name"], products_df["unit_price"]))
    product_names = products_df["name"].astype(str).tolist()
    imports: list[DayImport] = []
    for day_val, units in sorted(units_by_day.items()):
        day_sales = sum(units.get(n, 0) * int(price_map.get(n, 0)) for n in product_names)
        customers = max(100, int(sum(units.values()) / 0.12)) if sum(units.values()) > 0 else 0
        imports.append(DayImport(day_val, int(day_sales), customers, units))
    warnings.append("総客数・店舗総売上は、商品販売数から推定しています（売上サマリーCSV未使用）。")
    return imports, warnings


def prepare_square_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Square CSVの先頭メタ行をスキップし、ヘッダー行を自動検出する。"""
    if df.empty:
        return df

    cols = [str(c) for c in df.columns]
    if is_probable_header_row(cols):
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    # 先頭数行のどこかにヘッダーがあるパターン（Squareレポートで多い）
    scan_limit = min(25, len(df))
    for idx in range(scan_limit):
        row_vals = [str(v).strip() for v in df.iloc[idx].tolist()]
        if is_probable_header_row(row_vals):
            headers = [str(v).strip() for v in df.iloc[idx].tolist()]
            body = df.iloc[idx + 1 :].copy()
            body.columns = headers
            body = body.reset_index(drop=True)
            body.columns = [str(c).strip() for c in body.columns]
            keep_cols = [c for c in body.columns if str(c).strip()]
            return body[keep_cols]

    # Unnamed列のみの場合は1行目をヘッダーとして再構成
    if all(re.match(r"unnamed: \d+", normalize_col(c)) or str(c).isdigit() for c in cols):
        headers = [str(v).strip() for v in df.iloc[0].tolist()]
        body = df.iloc[1:].copy()
        body.columns = headers
        body = body.reset_index(drop=True)
        body.columns = [str(c).strip() for c in body.columns]
        return body

    return df


def build_datetime_series(work: pd.DataFrame, date_col: str, time_col: str | None) -> pd.Series:
    if time_col and time_col in work.columns:
        combined = work[date_col].astype(str).str.strip() + " " + work[time_col].astype(str).str.strip()
        return pd.to_datetime(combined, format="mixed", errors="coerce")
    return pd.to_datetime(work[date_col], format="mixed", errors="coerce")


def show_csv_debug_panel(raw_df: pd.DataFrame, mapping: dict[str, str | None] | None = None) -> None:
    with st.expander("📋 CSVヘッダー・先頭データ（デバッグ）", expanded=True):
        st.markdown("**CSVに含まれる列名（ヘッダー）**")
        st.code("\n".join(f"- {c}" for c in raw_df.columns.astype(str).tolist()), language="text")
        if mapping:
            st.markdown("**自動判別結果**")
            st.json({k: (v or "（未検出）") for k, v in mapping.items()})
        st.markdown("**先頭8行のデータ**")
        st.dataframe(raw_df.head(8), use_container_width=True)


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = decode_uploaded_bytes(raw)
    sep = detect_csv_separator(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
    return prepare_square_dataframe(df)


def read_uploaded_csv_raw(uploaded_file: Any) -> pd.DataFrame:
    """ヘッダー自動検出前の生データ（デバッグ表示用）。"""
    raw = uploaded_file.getvalue()
    text = decode_uploaded_bytes(raw)
    sep = detect_csv_separator(text)
    return pd.read_csv(io.StringIO(text), header=None, sep=sep, on_bad_lines="skip")


def match_product_name(raw_name: str, product_names: list[str]) -> str | None:
    name = str(raw_name).strip()
    if not name:
        return None
    if name in product_names:
        return name
    lower_map = {n.lower(): n for n in product_names}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    alias_entries: list[tuple[int, str, str]] = []
    for catalog, patterns in PRODUCT_NAME_ALIASES.items():
        if catalog not in product_names:
            continue
        for pattern in patterns:
            alias_entries.append((len(pattern), catalog, pattern))
    alias_entries.sort(key=lambda x: x[0], reverse=True)
    for _plen, catalog, pattern in alias_entries:
        if name == pattern or pattern in name:
            return catalog

    for pn in product_names:
        if pn in name or name in pn:
            return pn
    return None


def parse_square_csv(df: pd.DataFrame, products_df: pd.DataFrame) -> tuple[list[DayImport], list[str]]:
    """SquareエクスポートCSVを日次データへ変換する。"""
    warnings: list[str] = []
    df = prepare_square_dataframe(df)
    cols = [str(c) for c in df.columns.tolist()]
    product_names = products_df["name"].astype(str).tolist()

    # 横持ちマトリックス（日付が列・商品名が行）
    if is_matrix_format_df(df):
        return parse_matrix_format(df, products_df)

    date_col = find_column(cols, COLUMN_ALIASES["date"])
    if not date_col:
        raise ValueError(
            "日付列が見つかりません。CSVの列名をご確認ください。"
            "（対応例: 日付 / 時期 / Date / 取引日時、または日付が横並びのマトリックス形式）"
        )

    work = df.copy()
    time_col = find_column(cols, COLUMN_ALIASES["time"])
    work["_parsed_date"] = build_datetime_series(work, date_col, time_col)
    work = work.dropna(subset=["_parsed_date"])
    if work.empty:
        raise ValueError("有効な日付データがありません。")
    work["_day"] = work["_parsed_date"].dt.date

    item_col = find_column(cols, COLUMN_ALIASES["item"])
    qty_col = find_column(cols, COLUMN_ALIASES["quantity"])
    customers_col = find_column(cols, COLUMN_ALIASES["customers"])
    total_sales_col = find_column(cols, COLUMN_ALIASES["total_sales"])
    net_sales_col = find_column(cols, COLUMN_ALIASES["net_sales"])
    gross_sales_col = find_column(cols, COLUMN_ALIASES["gross_sales"])

    # 横持ち（商品名が列）形式の判定
    wide_product_cols = [c for c in cols if match_product_name(c, product_names)]
    imports: list[DayImport] = []

    if wide_product_cols and not item_col:
        for day_val, group in work.groupby("_day"):
            units: dict[str, int] = {}
            for col in wide_product_cols:
                matched = match_product_name(col, product_names)
                if matched:
                    units[matched] = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            day_sales = 0
            if total_sales_col:
                day_sales = int(pd.to_numeric(group[total_sales_col], errors="coerce").fillna(0).sum())
            elif net_sales_col:
                day_sales = int(pd.to_numeric(group[net_sales_col], errors="coerce").fillna(0).sum())
            else:
                day_sales = sum(units.get(n, 0) * int(products_df.loc[products_df["name"] == n, "unit_price"].iloc[0]) for n in units)
            customers = 0
            if customers_col:
                customers = int(pd.to_numeric(group[customers_col], errors="coerce").fillna(0).max())
            else:
                customers = max(100, int(sum(units.values()) / 0.12))
                warnings.append(f"{day_val}: 客数列が無いため推定値を使用しました。")
            imports.append(DayImport(day_val, day_sales, customers, units))
        return imports, warnings

    # 縦持ち（商品名×数量）形式
    if not item_col or not qty_col:
        raise ValueError(
            "商品別データ形式として認識できませんでした。"
            "「商品名」「数量」列、または商品名が列名の横持ちCSVをご利用ください。"
        )

    sales_col = net_sales_col or gross_sales_col
    for day_val, group in work.groupby("_day"):
        units: dict[str, int] = {}
        unmapped: list[str] = []
        for _, row in group.iterrows():
            matched = match_product_name(row[item_col], product_names)
            if not matched:
                unmapped.append(str(row[item_col]))
                continue
            qty = int(pd.to_numeric(row[qty_col], errors="coerce") or 0)
            units[matched] = units.get(matched, 0) + qty

        if unmapped:
            sample = ", ".join(sorted(set(unmapped))[:5])
            warnings.append(f"{day_val}: 未登録商品をスキップしました（例: {sample}）")

        if total_sales_col:
            day_sales = int(pd.to_numeric(group[total_sales_col], errors="coerce").fillna(0).max())
        elif sales_col:
            day_sales = int(pd.to_numeric(group[sales_col], errors="coerce").fillna(0).sum())
        else:
            price_map = dict(zip(products_df["name"], products_df["unit_price"]))
            day_sales = sum(units.get(n, 0) * int(price_map.get(n, 0)) for n in units)

        if customers_col:
            customers = int(pd.to_numeric(group[customers_col], errors="coerce").fillna(0).max())
        else:
            customers = max(100, int(sum(units.values()) / 0.12))
            warnings.append(f"{day_val}: 客数列が無いため推定値を使用しました。")

        imports.append(DayImport(day_val, day_sales, customers, units))

    return imports, warnings


# ---------------------------------------------------------------------------
# データ永続化
# ---------------------------------------------------------------------------
def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PRODUCTS_CSV.exists():
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

    if not DAILY_CSV.exists():
        init_empty_daily_csv()


def load_products() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").fillna(0).astype(int)
    return df


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


def bulk_import_days(imports: list[DayImport], products_df: pd.DataFrame, overwrite: bool = True) -> tuple[int, int]:
    """複数日分を一括取り込み。戻り値: (新規日数, 上書き日数)"""
    daily_df = load_daily_sales()
    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    created, updated = 0, 0

    active_products = products_df[products_df["is_active"] == 1].copy()
    if active_products.empty:
        active_products = products_df.copy()

    for day_data in imports:
        if day_data.day in existing_dates:
            if not overwrite:
                continue
            updated += 1
        else:
            created += 1
        units = {name: day_data.units_by_product.get(name, 0) for name in active_products["name"].astype(str)}
        save_daily_input(day_data.day, day_data.total_sales, day_data.total_customers, units, active_products)

    return created, updated


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


def add_product(name: str, unit_price: int) -> None:
    products_df = load_products()
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


def deactivate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 0
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


def activate_product(product_id: str) -> None:
    products_df = load_products()
    products_df.loc[products_df["product_id"] == product_id, "is_active"] = 1
    products_df.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")


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


def get_same_weekday_df(daily_df: pd.DataFrame, target_date: date, product_id: str) -> pd.DataFrame:
    ref_ts = to_ts(target_date)
    mask = (
        (daily_df["product_id"] == product_id)
        & (daily_df["date"].dt.weekday == ref_ts.weekday())
        & (daily_df["date"] < ref_ts)
    )
    return daily_df.loc[mask].copy()


def last_week_same_day_sales(daily_df: pd.DataFrame, product_id: str, ref: date) -> int:
    row = daily_df[(daily_df["date"] == to_ts(ref - timedelta(days=7))) & (daily_df["product_id"] == product_id)]
    return int(row["units_sold"].iloc[0]) if not row.empty else 0


def four_week_same_weekday_avg(daily_df: pd.DataFrame, product_id: str, ref: date) -> float:
    hist = get_same_weekday_df(daily_df, ref, product_id)
    last_4 = hist[hist["date"] >= to_ts(ref - timedelta(days=28))]
    return float(last_4["units_sold"].mean()) if not last_4.empty else 0.0


def calc_food_selection_rate(daily_df: pd.DataFrame, product_id: str, day_totals: pd.DataFrame, days: int = 7) -> float:
    if daily_df.empty or day_totals.empty:
        return 0.0
    end = to_ts(date.today())
    start = to_ts(date.today() - timedelta(days=days))
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
    fig.update_layout(title=f"{product_name} — 同曜日販売数の比較", yaxis_title="販売数（個）", template="plotly_white", height=360, showlegend=False)
    return fig


def plot_daily_trend(product_df: pd.DataFrame, product_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=product_df["date"], y=product_df["units_sold"], mode="lines+markers", name="販売個数", line=dict(color="#e94560", width=3))
    )
    fig.update_layout(title=f"{product_name} — 日次販売推移", xaxis_title="日付", yaxis_title="販売数（個）", template="plotly_white", height=400, hovermode="x unified")
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
    show_debug = st.checkbox("CSVヘッダーを確認（デバッグ）", value=False)

    if not uploaded_list:
        if is_daily_data_empty(daily_df):
            st.info(EMPTY_DATA_MESSAGE)
            st.markdown(
                "1. **商品別**の横持ちCSV（ワッフル・パフェ等の日別販売数）  \n"
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
        label = {"product_matrix": "商品別マトリックス", "sales_summary": "売上サマリー", "unknown": "未判定"}.get(kind, kind)
        st.write(f"- **{f.name}** → {label}")

    if show_debug:
        for f in files:
            st.markdown(f"**{f.name}**")
            prepared_df = prepare_square_dataframe(read_uploaded_csv_raw(f).copy())
            if not prepared_df.empty and not is_probable_header_row(list(prepared_df.columns.astype(str))):
                prepared_df = prepare_square_dataframe(prepared_df)
            mapping = detect_column_mapping([str(c) for c in prepared_df.columns.astype(str).tolist()])
            show_csv_debug_panel(prepared_df, mapping)

    try:
        imports, warnings = parse_dual_csv_upload(files, products_df)
    except Exception as exc:
        st.error(f"CSVの読み込みに失敗しました: {exc}")
        if show_debug and files:
            prepared_df = prepare_square_dataframe(read_uploaded_csv(files[0]))
            show_csv_debug_panel(prepared_df, detect_column_mapping(list(prepared_df.columns.astype(str))))
        return

    if not imports:
        st.warning("取り込み可能な日次データがありませんでした。")
        return

    preview_rows = []
    for d in imports:
        preview_rows.append(
            {
                "日付": d.day.strftime("%Y-%m-%d"),
                "店舗総売上": d.total_sales,
                "総客数": d.total_customers,
                "フード販売合計": sum(d.units_by_product.values()),
            }
        )
    st.markdown("#### 取り込みプレビュー")
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    existing_dates = set(daily_df["date"].dt.date.tolist()) if not daily_df.empty else set()
    overlap = [d.day for d in imports if d.day in existing_dates]
    if overlap:
        st.warning(f"既存データと重複する日付: {len(overlap)}日（上書き設定: {'ON' if overwrite else 'OFF'}）")

    for msg in warnings[:8]:
        st.caption(f"⚠ {msg}")
    if len(warnings) > 8:
        st.caption(f"…他 {len(warnings) - 8} 件")

    if st.button("データを一括取り込み", type="primary", use_container_width=True):
        created, updated = bulk_import_days(imports, products_df, overwrite=overwrite)
        st.success(f"取り込み完了: 新規 {created} 日 / 上書き {updated} 日")
        st.rerun()


def render_history_edit_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">過去データの確認・修正</p>', unsafe_allow_html=True)
    st.caption("CSV取り込み後の数値訂正や、1日分の手動追加ができます。")

    with st.expander("1日分を手動で追加・上書き"):
        active_df = products_df[products_df["is_active"] == 1].copy()
        if active_df.empty:
            st.warning("有効な商品がありません。")
        else:
            with st.form("manual_day_form"):
                target_date = st.date_input("日付", value=date.today(), key="manual_date")
                c1, c2 = st.columns(2)
                with c1:
                    total_sales = st.number_input("店舗総売上（円）", min_value=0, value=1_000_000, step=10_000)
                with c2:
                    total_customers = st.number_input("総客数（人）", min_value=0, value=2_000, step=10)
                units_by_product: dict[str, int] = {}
                for p in active_df.itertuples(index=False):
                    units_by_product[p.name] = int(
                        st.number_input(f"{p.name}（個）", min_value=0, value=80, step=1, key=f"manual_{p.product_id}")
                    )
                manual_submit = st.form_submit_button("この日のデータを保存", use_container_width=True)
            if manual_submit:
                if not daily_df.empty and (daily_df["date"] == to_ts(target_date)).any():
                    st.warning(f"{target_date} の既存データを上書き保存しました。")
                save_daily_input(target_date, int(total_sales), int(total_customers), units_by_product, active_df)
                st.success("保存しました。")
                st.rerun()

    if daily_df.empty:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    st.dataframe(make_daily_summary_table(daily_df), use_container_width=True, hide_index=True)
    available_dates = sorted(daily_df["date"].dt.date.unique().tolist(), reverse=True)
    selected_date = st.selectbox("修正する日付", available_dates, format_func=lambda d: d.strftime("%Y-%m-%d"))
    totals, units_map = get_daily_record_by_date(daily_df, selected_date)

    with st.form("history_edit_form"):
        c1, c2 = st.columns(2)
        with c1:
            edit_total_sales = st.number_input("店舗総売上（円）", min_value=0, value=int(totals.get("total_sales", 0)), step=10_000)
        with c2:
            edit_total_customers = st.number_input("総客数（人）", min_value=0, value=int(totals.get("total_customers", 0)), step=10)
        edit_units: dict[str, int] = {}
        for p in products_df.sort_values("product_id").itertuples(index=False):
            edit_units[p.name] = int(
                st.number_input(
                    f"{p.name}（個）",
                    min_value=0,
                    value=int(units_map.get(str(p.name), 0)),
                    step=1,
                    key=f"edit_{selected_date}_{p.product_id}",
                )
            )
        updated = st.form_submit_button("データを更新する（上書き保存）", type="primary", use_container_width=True)

    if updated:
        save_daily_input(selected_date, int(edit_total_sales), int(edit_total_customers), edit_units, products_df)
        st.success(f"{selected_date} のデータを更新しました。ダッシュボードに即反映されます。")
        st.rerun()


def render_product_tab(products_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">商品管理</p>', unsafe_allow_html=True)
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

    view_df = products_df.copy()
    view_df["状態"] = np.where(view_df["is_active"] == 1, "販売中", "販売終了")
    st.dataframe(view_df[["product_id", "name", "unit_price", "状態"]], use_container_width=True, hide_index=True)

    active_df = products_df[products_df["is_active"] == 1]
    inactive_df = products_df[products_df["is_active"] == 0]
    c1, c2 = st.columns(2)
    with c1:
        if not active_df.empty:
            selected_off = st.selectbox("販売終了", active_df["product_id"] + " | " + active_df["name"], key="deactivate_select")
            if st.button("販売終了にする", use_container_width=True):
                deactivate_product(selected_off.split(" | ")[0])
                st.rerun()
    with c2:
        if not inactive_df.empty:
            selected_on = st.selectbox("販売再開", inactive_df["product_id"] + " | " + inactive_df["name"], key="activate_select")
            if st.button("販売再開する", use_container_width=True):
                activate_product(selected_on.split(" | ")[0])
                st.rerun()


def render_dashboard_tab(products_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    st.markdown('<p class="section-title">ダッシュボード・可視化</p>', unsafe_allow_html=True)
    if daily_df.empty:
        st.warning(EMPTY_DATA_MESSAGE)
        return

    daily_df = daily_df.sort_values("date")
    day_totals = get_day_totals(daily_df)
    latest_date = day_totals["date"].max().date()

    selected_name = st.selectbox("対象フード商品", products_df["name"].tolist(), key="dashboard_product")
    selected = products_df[products_df["name"] == selected_name].iloc[0]
    product_id = selected["product_id"]
    unit_price = int(selected["unit_price"])

    product_df = daily_df[daily_df["product_id"] == product_id]
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
            <p>Square CSV取り込み + 手動修正 · 日商目安 ¥{DAILY_REVENUE_TARGET:,}+</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_daily_data_empty(daily_df):
        show_empty_data_notice()

    t1, t2, t3, t4 = st.tabs(
        ["Square CSVアップロード", "過去データ確認・修正", "商品管理", "ダッシュボード・可視化"]
    )
    with t1:
        render_square_upload_tab(products_df, daily_df)
    with t2:
        render_history_edit_tab(products_df, daily_df)
    with t3:
        render_product_tab(products_df)
    with t4:
        render_dashboard_tab(products_df, daily_df)


if __name__ == "__main__":
    main()
