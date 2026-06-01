"""
Square CSV ルートB: 商品別マトリックス + 売上サマリーの読み込み・日付結合
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "日付", "時期", "取引日", "取引日時", "営業日", "day", "売上日", "datetime"],
    "time": ["time", "時刻", "取引時刻", "時間"],
    "item": ["item", "item name", "商品名", "アイテム名", "品目", "品名", "メニュー", "商品"],
    "quantity": ["qty", "quantity", "数量", "販売数", "個数", "sold", "販売個数", "売上数量", "点数"],
    "net_sales": ["net sales", "ネット売上", "純売上", "net", "正味売上"],
    "gross_sales": ["gross sales", "総売上", "売上", "gross", "売上高", "総売上高"],
    "customers": ["customers", "客数", "来店客数", "取引数", "transactions", "総客数", "取引件数"],
    "total_sales": ["total sales", "店舗売上", "総売上高", "売上合計", "店舗総売上", "合計売上"],
}

MATRIX_ITEM_ALIASES = COLUMN_ALIASES["item"] + ["カテゴリ", "category", "種別", "分類"]
SKIP_ROW_KEYWORDS = ["合計", "総計", "小計", "total", "subtotal", "カテゴリ合計"]

VARIATION_COLUMN_ALIASES = [
    "商品バリエーション",
    "バリエーション",
    "variation",
    "item variation",
    "variant",
    "オプション",
]

# 単体では誤マッチしやすい Square 商品名（バリエーション列との組み合わせで判定）
GENERIC_ITEM_NAMES = {"パイ", "ケーキ", "コーヒー", "ラテ", "ティー"}

SKIP_VARIATION_VALUES = {"", "nan", "none", "定価", "regular", "default", "標準"}

PRODUCT_NAME_ALIASES: dict[str, list[str]] = {
    "パフェ": ["ワッフル パフェ", "ワッフル　パフェ"],
    "ワッフル": ["リエージュワッフル"],
    "抹茶ケーキ": ["抹茶チーズケーキ", "抹茶ケーキ", "抹茶 ケーキ"],
    "チーズケーキ": ["トンカチーズケーキ", "チーズ ケーキ"],
    "バナナパウンドケーキ": ["バナナパウンドケーキ", "バナナ パウンド"],
    "レモンケーキ": [
        "レモンパウンドケーキ",
        "レモンポピーシード パウンドケーキ",
        "レモン ケーキ",
        "レモンパウンド",
    ],
    # Square: 商品名=「パイ」+ バリエーションで区別
    "ミートパイ": ["パイ ミート", "ミートパイ", "ミート パイ", "パイ ミートパイ"],
    "レモンパイ": [
        "パイ レモンクリーム",
        "レモンクリーム",
        "レモンパイ",
        "レモン パイ",
        "パイ レモン",
        "レモンクリームパイ",
    ],
}

KV_CUSTOMER_METRIC_LABELS = ["売上取引履歴", "総売上取引履歴", "受取合計額の取引履歴"]
KV_SALES_METRIC_LABELS = ["純売上高", "総売上高", "売上合計", "受取合計額", "正味売上"]
KV_SUMMARY_LABEL_HINTS = KV_CUSTOMER_METRIC_LABELS + KV_SALES_METRIC_LABELS + [
    "総売上", "純売上", "取引履歴", "ギフトカード", "税金",
]

SUMMARY_CUSTOMER_ROW_KEYWORDS = ["取引", "transaction", "客数", "件数"]
SUMMARY_SALES_ROW_KEYWORDS = ["売上", "sales", "gross", "net", "収入", "合計金額"]


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


def normalize_col(name: str) -> str:
    s = str(name).replace("\ufeff", "").strip().lower()
    return re.sub(r"\s+", " ", s)


def find_column(columns: list[str], keys: list[str]) -> str | None:
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


def parse_column_as_date(col_name: str) -> date | None:
    s = str(col_name).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    if find_column([s], MATRIX_ITEM_ALIASES + COLUMN_ALIASES["quantity"]):
        return None
    parsed = pd.to_datetime(s, format="mixed", errors="coerce")
    if pd.isna(parsed) or parsed.year < 2000 or parsed.year > 2100:
        return None
    return parsed.date()


def get_matrix_date_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if parse_column_as_date(c) is not None]


def find_matrix_item_column(columns: list[str]) -> str | None:
    return find_column(columns, MATRIX_ITEM_ALIASES)


def is_matrix_format_df(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    return find_matrix_item_column(cols) is not None and len(get_matrix_date_columns(cols)) >= 1


def is_probable_matrix_header_row(row_vals: list[str]) -> bool:
    row_cols = [str(v).strip() for v in row_vals if str(v).strip() and str(v).lower() != "nan"]
    return find_matrix_item_column(row_cols) is not None and len(get_matrix_date_columns(row_cols)) >= 2


def is_probable_header_row(row_vals: list[str]) -> bool:
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
        return "\t" if line.count("\t") >= line.count(",") else ","
    return ","


def parse_yen_to_int(val: Any) -> int:
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-", "—"):
        return 0
    negative = "(" in s or s.startswith("-")
    s = re.sub(r"[￥¥,\s\"']", "", s).replace("(", "").replace(")", "")
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
    if cell_value_is_yen(val):
        return parse_yen_to_int(val), True
    return int(pd.to_numeric(val, errors="coerce") or 0), False


def normalize_product_label(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"[（(][^）)]*[）)]", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def find_variation_column(columns: list[str]) -> str | None:
    return find_column(columns, VARIATION_COLUMN_ALIASES)


def build_square_product_label(item_name: str, variation: str = "") -> str:
    """Squareの「商品名」+「バリエーション」を1つの照合用文字列にする。"""
    item = str(item_name).strip()
    var = str(variation).strip()
    if not item or item.lower() == "nan":
        return ""
    if not var or var.lower() in SKIP_VARIATION_VALUES:
        return item
    return f"{item} {var}"


def match_product_name(raw_name: str, product_names: list[str]) -> str | None:
    name = str(raw_name).strip()
    if not name:
        return None
    norm_name = normalize_product_label(name)
    if name in product_names:
        return name
    norm_to_catalog = {normalize_product_label(n): n for n in product_names if normalize_product_label(n)}
    if norm_name in norm_to_catalog:
        return norm_to_catalog[norm_name]

    alias_entries: list[tuple[int, str, str]] = []
    for catalog, patterns in PRODUCT_NAME_ALIASES.items():
        if catalog not in product_names:
            continue
        for pattern in patterns:
            alias_entries.append((len(pattern), catalog, pattern))
    alias_entries.sort(key=lambda x: x[0], reverse=True)
    for _plen, catalog, pattern in alias_entries:
        norm_pattern = normalize_product_label(pattern)
        if name == pattern or pattern in name or (norm_pattern and norm_pattern in norm_name):
            return catalog

    item_only = name.split()[0] if " " not in name else None
    for pn in sorted(product_names, key=len, reverse=True):
        norm_pn = normalize_product_label(pn)
        if pn in name or (norm_pn and norm_pn in norm_name):
            return pn
        # 「パイ」→「ミートパイ」誤爆を防ぐ: 短いCSV名が長い登録名に含まれるだけの一致は禁止
        if name in pn and item_only not in GENERIC_ITEM_NAMES and len(name) >= 4:
            return pn
    return None


def prepare_square_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [str(c) for c in df.columns]
    if is_probable_header_row(cols):
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out
    scan_limit = min(25, len(df))
    for idx in range(scan_limit):
        row_vals = [str(v).strip() for v in df.iloc[idx].tolist()]
        if is_probable_header_row(row_vals):
            headers = [str(v).strip() for v in df.iloc[idx].tolist()]
            body = df.iloc[idx + 1 :].copy()
            body.columns = headers
            body = body.reset_index(drop=True)
            body.columns = [str(c).strip() for c in body.columns]
            return body[[c for c in body.columns if str(c).strip()]]
    if all(re.match(r"unnamed: \d+", normalize_col(c)) or str(c).isdigit() for c in cols):
        headers = [str(v).strip() for v in df.iloc[0].tolist()]
        body = df.iloc[1:].copy()
        body.columns = headers
        body = body.reset_index(drop=True)
        body.columns = [str(c).strip() for c in body.columns]
        return body
    return df


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = decode_uploaded_bytes(raw)
    sep = detect_csv_separator(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
    return prepare_square_dataframe(df)


def normalize_square_kv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] < 2:
        return df
    if "metric" in df.columns and "value" in df.columns:
        return df
    return pd.DataFrame(
        {"metric": df.iloc[:, 0].astype(str).str.strip(), "value": df.iloc[:, 1]}
    )


def read_square_sales_summary_csv(uploaded_file: Any) -> pd.DataFrame:
    text = decode_uploaded_bytes(uploaded_file.getvalue())
    df = pd.read_csv(io.StringIO(text))
    return normalize_square_kv_dataframe(df)


def kv_metric_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "metric" in df.columns:
        return df["metric"].astype(str).str.strip(), df["value"]
    return df.iloc[:, 0].astype(str).str.strip(), df.iloc[:, 1]


def is_square_kv_summary_df(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] < 2:
        return False
    if get_matrix_date_columns([str(c) for c in df.columns]):
        return False
    labels, _ = kv_metric_series(df)
    hits = sum(1 for lab in labels if any(h in lab for h in KV_SUMMARY_LABEL_HINTS))
    return hits >= 2


def is_customer_metric_label(label: str) -> bool:
    text = str(label).strip().lower()
    if not text or is_summary_row_label(label):
        return False
    if any(k in text for k in SUMMARY_SALES_ROW_KEYWORDS) and "取引" not in text:
        return False
    return any(k in text for k in SUMMARY_CUSTOMER_ROW_KEYWORDS)


def is_sales_metric_label(label: str) -> bool:
    text = str(label).strip().lower()
    if not text or is_summary_row_label(label) or is_customer_metric_label(label):
        return False
    return any(k in text for k in SUMMARY_SALES_ROW_KEYWORDS)


def matrix_to_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    item_col = find_matrix_item_column(cols)
    var_col = find_variation_column(cols)
    date_cols = get_matrix_date_columns(cols)
    if not item_col or not date_cols:
        raise ValueError("マトリックス形式の列構成を認識できませんでした。")
    id_cols = [item_col] + ([var_col] if var_col else [])
    long_df = df[id_cols + date_cols].copy()
    long_df = long_df.melt(id_vars=id_cols, value_vars=date_cols, var_name="_date_col", value_name="quantity")
    long_df["date"] = long_df["_date_col"].apply(lambda x: parse_column_as_date(str(x)))
    long_df = long_df.dropna(subset=["date"])
    if var_col:
        long_df["product_name"] = long_df.apply(
            lambda r: build_square_product_label(r[item_col], r[var_col]),
            axis=1,
        )
    else:
        long_df["product_name"] = long_df[item_col].astype(str).str.strip()
    parsed = long_df["quantity"].apply(parse_matrix_cell)
    long_df["quantity"] = parsed.apply(lambda x: x[0])
    long_df["is_revenue"] = parsed.apply(lambda x: x[1])
    long_df = long_df[long_df["product_name"].astype(str).str.len() > 0]
    long_df = long_df[~long_df["product_name"].apply(is_summary_row_label)]
    return long_df[["date", "product_name", "quantity", "is_revenue"]]


def summarize_square_row_mapping(df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """CSV各行がどの登録商品に紐づくか一覧（取り込み前の確認用）。"""
    prepared = prepare_square_dataframe(df)
    if not is_matrix_format_df(prepared):
        return pd.DataFrame()
    product_names = products_df["name"].astype(str).tolist()
    cols = [str(c) for c in prepared.columns]
    item_col = find_matrix_item_column(cols)
    var_col = find_variation_column(cols)
    date_cols = get_matrix_date_columns(cols)
    rows_out: list[dict[str, Any]] = []
    for _, row in prepared.iterrows():
        item = str(row[item_col]).strip()
        if not item or is_summary_row_label(item):
            continue
        variation = str(row[var_col]).strip() if var_col else ""
        label = build_square_product_label(item, variation)
        matched = match_product_name(label, product_names)
        period_total = 0
        for dcol in date_cols:
            raw, is_rev = parse_matrix_cell(row[dcol])
            period_total += raw
        if period_total == 0:
            continue
        rows_out.append(
            {
                "Square商品名": item,
                "バリエーション": variation if variation and variation.lower() not in SKIP_VARIATION_VALUES else "—",
                "照合キー": label,
                "アプリ商品": matched or "（未紐づけ）",
                "期間合計": period_total,
            }
        )
    return pd.DataFrame(rows_out)


def parse_matrix_units_by_day(
    df: pd.DataFrame, products_df: pd.DataFrame
) -> tuple[dict[date, dict[str, int]], dict[date, float], list[str]]:
    warnings = ["商品別マトリックスCSVを読み込みました。"]
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
    unmapped_labels: set[str] = set()

    for day_val, day_group in long_df.groupby("date"):
        units = {name: 0 for name in product_names}
        day_revenue = 0.0
        for _, row in day_group.iterrows():
            label = str(row["product_name"])
            matched = match_product_name(label, product_names)
            if not matched:
                if int(row["quantity"]) != 0:
                    unmapped_labels.add(label)
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
        units_by_day[day_val] = units
        revenue_by_day[day_val] = max(day_revenue, 1.0) if sum(units.values()) > 0 else 0.0

    if used_yen:
        warnings.append("セルが金額（¥）表記のため、登録単価から販売数を換算しました。")
    if unmapped_labels:
        sample = "、".join(sorted(unmapped_labels)[:6])
        warnings.append(f"未紐づけのSquare行: {sample}（他{max(0, len(unmapped_labels) - 6)}件）")
    return units_by_day, revenue_by_day, warnings


def build_datetime_series(work: pd.DataFrame, date_col: str, time_col: str | None) -> pd.Series:
    if time_col and time_col in work.columns:
        combined = work[date_col].astype(str).str.strip() + " " + work[time_col].astype(str).str.strip()
        return pd.to_datetime(combined, format="mixed", errors="coerce")
    return pd.to_datetime(work[date_col], format="mixed", errors="coerce")


def parse_sales_summary_vertical(df: pd.DataFrame) -> tuple[dict[date, DaySummary], list[str]]:
    warnings = ["売上サマリーCSV（日別・縦持ち）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    cols = [str(c) for c in prepared.columns]
    date_col = find_column(cols, COLUMN_ALIASES["date"])
    if not date_col:
        raise ValueError("売上サマリーCSVに日付列が見つかりません。")
    customers_col = find_column(cols, COLUMN_ALIASES["customers"])
    sales_col = (
        find_column(cols, COLUMN_ALIASES["total_sales"])
        or find_column(cols, COLUMN_ALIASES["gross_sales"])
        or find_column(cols, COLUMN_ALIASES["net_sales"])
    )
    work = prepared.copy()
    work["_parsed_date"] = build_datetime_series(work, date_col, find_column(cols, COLUMN_ALIASES["time"]))
    work = work.dropna(subset=["_parsed_date"])
    work["_day"] = work["_parsed_date"].dt.date
    by_day: dict[date, DaySummary] = {}
    for day_val, group in work.groupby("_day"):
        customers = int(pd.to_numeric(group[customers_col], errors="coerce").fillna(0).max()) if customers_col else 0
        sales = 0
        if sales_col:
            for val in group[sales_col]:
                sales = max(sales, parse_yen_to_int(val) if cell_value_is_yen(val) else int(pd.to_numeric(val, errors="coerce") or 0))
        if customers == 0 and sales == 0:
            continue
        by_day[day_val] = DaySummary(total_sales=sales, total_customers=customers)
    if not by_day:
        raise ValueError("売上サマリーCSVから有効な日次データを抽出できませんでした。")
    return by_day, warnings


def parse_sales_summary_matrix(df: pd.DataFrame) -> tuple[dict[date, DaySummary], list[str]]:
    warnings = ["売上サマリーCSV（日別・マトリックス）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    item_col = find_matrix_item_column([str(c) for c in prepared.columns])
    date_cols = get_matrix_date_columns([str(c) for c in prepared.columns])
    if not item_col or not date_cols:
        raise ValueError("売上サマリーCSVの列構成を認識できませんでした。")
    by_day: dict[date, DaySummary] = {}
    for dcol in date_cols:
        if day_val := parse_column_as_date(dcol):
            by_day[day_val] = DaySummary(0, 0)
    for _, row in prepared.iterrows():
        label = str(row[item_col]).strip()
        if not label or is_summary_row_label(label):
            continue
        for dcol in date_cols:
            day_val = parse_column_as_date(dcol)
            if not day_val:
                continue
            val_raw = row[dcol]
            val = parse_yen_to_int(val_raw) if cell_value_is_yen(val_raw) else int(pd.to_numeric(val_raw, errors="coerce") or 0)
            current = by_day.setdefault(day_val, DaySummary(0, 0))
            if is_customer_metric_label(label):
                by_day[day_val] = DaySummary(current.total_sales, max(current.total_customers, val))
            elif is_sales_metric_label(label):
                by_day[day_val] = DaySummary(max(current.total_sales, val), current.total_customers)
    by_day = {d: s for d, s in by_day.items() if s.total_sales > 0 or s.total_customers > 0}
    if not by_day:
        raise ValueError("売上サマリーCSVから日別の総売上・取引件数を抽出できませんでした。")
    return by_day, warnings


def extract_period_summary_from_kv(df: pd.DataFrame) -> tuple[int, int]:
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
        raise ValueError("売上サマリーCSVから期間合計を抽出できませんでした。")
    return customers, sales


def distribute_period_summary_to_days(
    period_customers: int, period_sales: int, daily_weights: dict[date, float]
) -> dict[date, DaySummary]:
    if not daily_weights:
        raise ValueError("商品別CSVの日付が無いため、期間サマリーを日別に配分できません。")
    total_w = sum(daily_weights.values()) or 1.0
    by_day: dict[date, DaySummary] = {}
    allocated_sales = allocated_customers = 0
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
        by_day[day_val] = DaySummary(max(0, day_sales), max(0, day_customers))
    return by_day


def parse_sales_summary_csv(
    df: pd.DataFrame, daily_weights: dict[date, float] | None = None
) -> tuple[dict[date, DaySummary], list[str]]:
    df = normalize_square_kv_dataframe(df)
    if is_square_kv_summary_df(df):
        if not daily_weights:
            raise ValueError("期間集計の売上サマリーは、商品別CSVと一緒にアップロードしてください。")
        period_c, period_s = extract_period_summary_from_kv(df)
        warnings = ["売上サマリー（期間集計）を商品別売上比率で日別配分しました。"]
        return distribute_period_summary_to_days(period_c, period_s, daily_weights), warnings

    prepared = prepare_square_dataframe(df)
    if is_matrix_format_df(prepared):
        item_col = find_matrix_item_column([str(c) for c in prepared.columns])
        if item_col and any(
            is_customer_metric_label(str(row[item_col])) or is_sales_metric_label(str(row[item_col]))
            for _, row in prepared.iterrows()
        ):
            return parse_sales_summary_matrix(prepared)
    return parse_sales_summary_vertical(prepared)


def classify_uploaded_csv(df: pd.DataFrame, products_df: pd.DataFrame, filename: str = "") -> str:
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
        product_rows = metric_rows = 0
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
    sales_col = find_column(cols, COLUMN_ALIASES["total_sales"]) or find_column(cols, COLUMN_ALIASES["net_sales"])
    customers_col = find_column(cols, COLUMN_ALIASES["customers"])
    if date_col and (sales_col or customers_col):
        return "sales_summary"
    if date_col and find_column(cols, COLUMN_ALIASES["item"]):
        return "product_matrix"
    return "unknown"


def merge_dual_csv_imports(
    units_by_day: dict[date, dict[str, int]],
    summary_by_day: dict[date, DaySummary],
    products_df: pd.DataFrame,
) -> tuple[list[DayImport], list[str]]:
    warnings = ["ルートB: 2枚のCSVを日付で結合しました。"]
    product_names = products_df["name"].astype(str).tolist()
    price_map = dict(zip(products_df["name"], products_df["unit_price"]))
    all_days = sorted(set(units_by_day) | set(summary_by_day))
    if not all_days:
        raise ValueError("結合できる日付データがありません。")
    only_products = set(units_by_day) - set(summary_by_day)
    only_summary = set(summary_by_day) - set(units_by_day)
    if only_products:
        warnings.append(f"売上サマリーに無い日付: {len(only_products)}日（販売数のみ・客数/売上は推定）。")
    if only_summary:
        warnings.append(f"商品別CSVに無い日付: {len(only_summary)}日（客数/売上のみ）。")
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
            est = int(sum(units.get(n, 0) * int(price_map.get(n, 0)) for n in product_names))
            if total_sales == 0 and est > 0:
                total_sales = est
        if not summary or total_customers == 0:
            if sum(units.values()) > 0:
                total_customers = max(total_customers, int(sum(units.values()) / 0.12))
        imports.append(DayImport(day_val, total_sales, total_customers, units))
    return imports, warnings


def load_uploaded_dataframe(uploaded_file: Any, products_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    name = getattr(uploaded_file, "name", "") or ""
    name_lower = name.lower()
    if "sales-summary" in name_lower or ("売上サマリー" in name and "商品" not in name):
        df = read_square_sales_summary_csv(uploaded_file)
    else:
        df = read_uploaded_csv(uploaded_file)
    return classify_uploaded_csv(df, products_df, name), df


def parse_dual_csv_upload(uploaded_files: list[Any], products_df: pd.DataFrame) -> tuple[list[DayImport], list[str]]:
    if len(uploaded_files) > 2:
        raise ValueError("アップロードは2ファイルまでです。")
    classified: list[tuple[str, pd.DataFrame, str]] = []
    for f in uploaded_files:
        kind, df = load_uploaded_dataframe(f, products_df)
        classified.append((f.name, df, kind))
    if len(classified) == 1:
        _fname, df, kind = classified[0]
        if kind == "product_matrix":
            units, weights, w = parse_matrix_units_by_day(df, products_df)
            product_names = products_df["name"].astype(str).tolist()
            price_map = dict(zip(products_df["name"], products_df["unit_price"]))
            imports = [
                DayImport(
                    d,
                    int(sum(units[d].get(n, 0) * int(price_map.get(n, 0)) for n in product_names)),
                    max(100, int(sum(units[d].values()) / 0.12)) if sum(units[d].values()) > 0 else 0,
                    units[d],
                )
                for d in sorted(units)
            ]
            return imports, w + ["商品別CSVのみ。客数・店舗売上は推定値です。"]
        if kind == "sales_summary":
            raise ValueError("売上サマリー1枚のみの場合は、商品別マトリックスCSVも必要です。")
        raise ValueError(f"CSV形式を認識できませんでした: {classified[0][0]}")
    product_file = next((x for x in classified if x[2] == "product_matrix"), None)
    summary_file = next((x for x in classified if x[2] == "sales_summary"), None)
    if not product_file or not summary_file:
        kinds = ", ".join(f"{x[0]}→{x[2]}" for x in classified)
        raise ValueError(f"2枚の内訳を認識できませんでした。（判定: {kinds}）")
    units_by_day, revenue_by_day, w1 = parse_matrix_units_by_day(product_file[1], products_df)
    summary_by_day, w2 = parse_sales_summary_csv(summary_file[1], daily_weights=revenue_by_day)
    imports, w3 = merge_dual_csv_imports(units_by_day, summary_by_day, products_df)
    return imports, w1 + w2 + w3
