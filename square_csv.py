"""
Square CSV ルートB: 商品別マトリックス + 売上サマリーの読み込み・日付結合
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "日付", "時期", "取引日", "取引日時", "営業日", "day", "売上日", "datetime"],
    "time": ["time", "時刻", "取引時刻", "時間"],
    "item": ["item", "item name", "商品名", "アイテム名", "品目", "品名", "メニュー", "商品"],
    "quantity": ["qty", "quantity", "数量", "販売数", "個数", "sold", "販売個数", "売上数量", "点数"],
    "net_sales": ["net sales", "ネット売上", "純売上", "純売上高", "net", "正味売上"],
    "gross_sales": ["gross sales", "総売上", "売上", "gross", "売上高", "総売上高"],
    "customers": ["customers", "客数", "来店客数", "取引数", "transactions", "総客数", "取引件数"],
    "total_sales": ["total sales", "店舗売上", "総売上高", "売上合計", "店舗総売上", "合計売上"],
}

MATRIX_ITEM_ALIASES = COLUMN_ALIASES["item"] + ["カテゴリ", "category", "種別", "分類"]
SUMMARY_LABEL_COLUMN_ALIASES = ["レポート日", "report date", "指標", "項目", "metric"]
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
# 店舗総売上・単品売上高の計算は税抜の純売上高を優先
KV_SALES_METRIC_PRIORITY = [
    "純売上高",
    "正味売上",
    "ネット売上",
    "net sales",
]
KV_SALES_METRIC_FALLBACK = [
    "受取合計額",
    "総売上高",
    "売上合計",
    "総売上",
    "売上合計額",
    "gross sales",
    "total sales",
]
KV_SALES_METRIC_LABELS = KV_SALES_METRIC_PRIORITY + KV_SALES_METRIC_FALLBACK
KV_SUMMARY_LABEL_HINTS = KV_CUSTOMER_METRIC_LABELS + KV_SALES_METRIC_LABELS + [
    "総売上", "純売上", "取引履歴", "ギフトカード", "税金",
]

SUMMARY_CUSTOMER_ROW_KEYWORDS = ["取引", "transaction", "客数", "件数"]
SUMMARY_SALES_ROW_KEYWORDS = ["売上", "sales", "gross", "net", "収入", "合計金額"]

# これ未満の日次は「実質データなし」とみなし取り込みしない（配分の端数などを除外）
MIN_MEANINGFUL_DAY_SALES = 100
MIN_MEANINGFUL_DAY_CUSTOMERS = 5


@dataclass
class DayImport:
    day: date
    total_sales: int
    total_customers: int
    units_by_product: dict[str, int]
    product_sales_by_product: dict[str, int] | None = None


@dataclass
class DaySummary:
    total_sales: int
    total_customers: int


def is_meaningful_day(
    units: dict[str, int],
    summary: DaySummary | None = None,
) -> bool:
    """フード販売か店舗サマリーに実データがある日だけ残す。"""
    if sum(units.values()) > 0:
        return True
    if not summary:
        return False
    return (
        summary.total_sales >= MIN_MEANINGFUL_DAY_SALES
        or summary.total_customers >= MIN_MEANINGFUL_DAY_CUSTOMERS
    )


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
            elif len(col_norm) >= 3:
                if col_norm.startswith(key) or key.startswith(col_norm):
                    score = 800 - priority
                elif key in col_norm:
                    score = 600 - priority
                elif len(col_norm) >= 4 and col_norm in key:
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
    return find_column(columns, MATRIX_ITEM_ALIASES + SUMMARY_LABEL_COLUMN_ALIASES)


def resolve_summary_label_column(columns: list[str]) -> str | None:
    """売上サマリー日次マトリックスの指標名列（先頭列フォールバック）。"""
    label_col = find_matrix_item_column(columns)
    if label_col:
        return label_col
    date_cols = set(get_matrix_date_columns(columns))
    for col in columns:
        if col not in date_cols:
            return col
    return None


def is_net_sales_metric_label(label: str) -> bool:
    lab = str(label).strip()
    if not lab or is_customer_metric_label(lab):
        return False
    return lab == "純売上高" or ("純売上高" in lab and "取引" not in lab)


def is_sales_summary_daily_matrix_df(df: pd.DataFrame) -> bool:
    """売上サマリー（日次）: 列が日付・行が純売上高などのマトリックス。"""
    if df.empty:
        return False
    cols = [str(c) for c in df.columns]
    if len(get_matrix_date_columns(cols)) < 1:
        return False
    label_col = resolve_summary_label_column(cols)
    if not label_col:
        return False
    for _, row in df.iterrows():
        if is_net_sales_metric_label(str(row[label_col])):
            return True
    return False


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


def is_known_square_metric_label(name: str) -> bool:
    """Square売上サマリーで使われる指標行（「合計」を含むがスキップしない）。"""
    lab = str(name).strip()
    if not lab:
        return False
    lower = lab.lower()
    for key in KV_SALES_METRIC_PRIORITY + KV_SALES_METRIC_FALLBACK:
        if key.lower() in lower or key in lab:
            return True
    return any(key in lab for key in KV_CUSTOMER_METRIC_LABELS)


def is_summary_row_label(name: str) -> bool:
    lab = str(name).strip()
    if not lab:
        return True
    if is_known_square_metric_label(lab):
        return False
    label = lab.lower()
    return any(k in label for k in SKIP_ROW_KEYWORDS)


def decode_uploaded_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    sample = raw[:800]
    if sample and sample.count(b"\x00") > len(sample) // 4:
        return raw.decode("utf-16-le")
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def detect_csv_separator(text: str) -> str:
    for line in text.splitlines():
        if not line.strip():
            continue
        tabs = line.count("\t")
        commas = line.count(",")
        if tabs == 0 and commas == 0:
            continue
        return "\t" if tabs > commas else ","
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


_CUSTOM_PRODUCT_MAPPINGS: dict[str, str] = {}


def set_custom_product_mappings(mappings: dict[str, str] | None) -> None:
    """square_label → アプリ登録商品名 のユーザー定義マッピング。"""
    global _CUSTOM_PRODUCT_MAPPINGS
    _CUSTOM_PRODUCT_MAPPINGS = dict(mappings or {})


def get_custom_product_mappings() -> dict[str, str]:
    return dict(_CUSTOM_PRODUCT_MAPPINGS)


def resolve_custom_mapping(raw_name: str, product_names: list[str]) -> str | None:
    """ユーザー紐づけ設定を最優先で適用。"""
    name = str(raw_name).strip()
    if not name or not _CUSTOM_PRODUCT_MAPPINGS:
        return None
    norm_name = normalize_product_label(name)
    if name in _CUSTOM_PRODUCT_MAPPINGS:
        target = _CUSTOM_PRODUCT_MAPPINGS[name]
        if target in product_names:
            return target
    for key, target in _CUSTOM_PRODUCT_MAPPINGS.items():
        if normalize_product_label(key) == norm_name and target in product_names:
            return target
    return None


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
    custom = resolve_custom_mapping(name, product_names)
    if custom:
        return custom
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


def has_vertical_product_columns(columns: list[str]) -> bool:
    """商品売上サマリー（縦持ち・期間集計）の列が既に揃っている。"""
    item_col = find_column(columns, COLUMN_ALIASES["item"])
    qty_col = find_column(columns, ["販売数", "販売商品数"] + COLUMN_ALIASES["quantity"])
    return bool(item_col and qty_col and not get_matrix_date_columns(columns))


def prepare_square_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [str(c) for c in df.columns]
    if has_vertical_product_columns(cols):
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out
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


def fix_square_daily_summary_csv_text(text: str) -> str:
    """「売上サマリー - 日次」で1行目が改行含みタイトルになるSquare CSVを補正。"""
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    first = lines[0].strip().strip('"')
    second = lines[1].strip()
    if "売上サマリー" in first and "日次" in first and "レポート日" in second:
        return "\n".join(lines[1:])
    return text


def fix_square_sales_summary_csv_text(text: str) -> str:
    """期間サマリーCSVのタイトル行を除き、指標行から読み始める。"""
    text = fix_square_daily_summary_csv_text(text)
    lines = text.splitlines()
    if lines:
        head = lines[0].strip()
        if "レポート日" in head and re.search(r"\d{4}[-/]\d", head):
            return text
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if re.match(r'^"[^"]+","', s):
            return "\n".join(lines[idx:])
        if "," in s and any(h in s for h in KV_SUMMARY_LABEL_HINTS):
            return "\n".join(lines[idx:])
    return text


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = fix_square_sales_summary_csv_text(decode_uploaded_bytes(raw))
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
    """売上サマリー（期間KV / 日次マトリックス両対応）。"""
    return read_uploaded_csv(uploaded_file)


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


def sales_metric_priority_score(label: str) -> int:
    """大きいほど税抜の店舗売上（純売上系）に近い行。"""
    lab = str(label).strip()
    if not lab or is_summary_row_label(lab) or is_customer_metric_label(lab):
        return -1
    lower = lab.lower()
    for idx, key in enumerate(KV_SALES_METRIC_PRIORITY):
        if key.lower() in lower or key in lab:
            return 200 - idx
    for idx, key in enumerate(KV_SALES_METRIC_FALLBACK):
        if key.lower() in lower or key in lab:
            return 80 - idx
    if is_sales_metric_label(lab):
        return 30
    return -1


def pick_net_sales_from_kv(metric_map: dict[str, Any]) -> int:
    """売上サマリーKVから純売上高（税抜）のみを取得。"""
    for key, val in metric_map.items():
        lab = str(key).strip()
        if lab == "純売上高" or lab.endswith("純売上高"):
            return parse_yen_to_int(val)
    for key, val in metric_map.items():
        if "純売上高" in str(key) and "取引" not in str(key):
            return parse_yen_to_int(val)
    return pick_sales_amount_from_metric_map(metric_map)


def pick_gross_sales_from_kv(metric_map: dict[str, Any]) -> int:
    """期間の総売上高（配分比率用）。"""
    for key in ("総売上高", "総売上", "商品"):
        if key in metric_map:
            return parse_yen_to_int(metric_map[key])
    for key, val in metric_map.items():
        if "総売上高" in str(key) and "取引" not in str(key):
            return parse_yen_to_int(val)
    return 0


def pick_sales_amount_from_metric_map(metric_map: dict[str, Any]) -> int:
    best_score = -1
    best_val = 0
    for label, val in metric_map.items():
        score = sales_metric_priority_score(label)
        if score < 0:
            continue
        amount = parse_yen_to_int(val) if not isinstance(val, (int, float)) else int(val)
        if score > best_score:
            best_score = score
            best_val = amount
    return best_val


def pick_preferred_sales_column(columns: list[str]) -> str | None:
    scored: list[tuple[int, str]] = []
    for col in columns:
        score = sales_metric_priority_score(col)
        if score >= 0:
            scored.append((score, col))
    if not scored:
        return (
            find_column(columns, COLUMN_ALIASES["total_sales"])
            or find_column(columns, COLUMN_ALIASES["gross_sales"])
            or find_column(columns, COLUMN_ALIASES["net_sales"])
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


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


def parse_date_range_from_filename(filename: str) -> tuple[date, date] | None:
    """例: 商品売上サマリー-2026-05-01-2026-06-01.csv"""
    m = re.search(
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[-_](\d{4})[-/](\d{1,2})[-/](\d{1,2})",
        filename,
    )
    if not m:
        return None
    y1, mo1, d1, y2, mo2, d2 = (int(x) for x in m.groups())
    return date(y1, mo1, d1), date(y2, mo2, d2)


def parse_date_range_from_filenames(*filenames: str) -> tuple[date, date] | None:
    for name in filenames:
        if parsed := parse_date_range_from_filename(name):
            return parsed
    return None


def days_in_range(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        return []
    days: list[date] = []
    cur = start_date
    while cur <= end_date:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def is_product_period_summary_df(df: pd.DataFrame) -> bool:
    """商品売上サマリー（期間集計・縦持ち）。"""
    if df.empty:
        return False
    cols = [str(c) for c in df.columns]
    if get_matrix_date_columns(cols):
        return False
    if is_sales_summary_daily_matrix_df(df):
        return False
    item_col = find_column(cols, COLUMN_ALIASES["item"])
    qty_col = find_column(cols, ["販売数", "販売商品数"] + COLUMN_ALIASES["quantity"])
    return bool(item_col and qty_col)


def allocate_integer_by_weights(total: int, weights: dict[date, float]) -> dict[date, int]:
    if total <= 0 or not weights:
        return {d: 0 for d in weights}
    wsum = sum(weights.values()) or 1.0
    raw = {d: total * weights[d] / wsum for d in weights}
    floored = {d: int(v) for d, v in raw.items()}
    remainder = total - sum(floored.values())
    if remainder > 0:
        for d in sorted(weights, key=lambda x: raw[x] - floored[x], reverse=True)[:remainder]:
            floored[d] += 1
    elif remainder < 0:
        for d in sorted(weights, key=lambda x: raw[x] - floored[x])[: -remainder]:
            floored[d] -= 1
    return floored


def parse_product_period_summary(
    df: pd.DataFrame, products_df: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """期間集計の商品売上サマリー → 商品ごとの期間合計販売数・純売上。"""
    warnings = ["商品売上サマリー（期間集計）を読み込みました。"]
    prepared = prepare_square_dataframe(df)
    cols = [str(c) for c in prepared.columns]
    item_col = find_column(cols, COLUMN_ALIASES["item"])
    var_col = find_variation_column(cols)
    qty_col = find_column(cols, ["販売数", "販売商品数"] + COLUMN_ALIASES["quantity"])
    sales_col = find_column(cols, COLUMN_ALIASES["net_sales"])
    if not item_col or not qty_col:
        raise ValueError("商品売上サマリーCSVの列（商品名・販売数）を認識できませんでした。")

    product_names = products_df["name"].astype(str).tolist()
    period_units = {name: 0 for name in product_names}
    period_sales = {name: 0 for name in product_names}
    unmapped: set[str] = set()

    for _, row in prepared.iterrows():
        item = str(row[item_col]).strip()
        if not item or is_summary_row_label(item):
            continue
        variation = str(row[var_col]).strip() if var_col else ""
        label = build_square_product_label(item, variation)
        matched = match_product_name(label, product_names)
        if not matched:
            qty_val = int(pd.to_numeric(row[qty_col], errors="coerce") or 0)
            if qty_val > 0:
                unmapped.add(label)
            continue
        qty_val = int(pd.to_numeric(row[qty_col], errors="coerce") or 0)
        line_sales = parse_yen_to_int(row[sales_col]) if sales_col else 0
        period_units[matched] += qty_val
        period_sales[matched] += line_sales

    if unmapped:
        sample = "、".join(sorted(unmapped)[:6])
        warnings.append(f"未紐づけのSquare行: {sample}（他{max(0, len(unmapped) - 6)}件）")
    if sum(period_units.values()) <= 0:
        raise ValueError("商品売上サマリーから登録フードの販売数を抽出できませんでした。")
    return period_units, period_sales, warnings


def distribute_period_products_to_days(
    period_units: dict[str, int],
    period_sales: dict[str, int],
    summary_by_day: dict[date, DaySummary],
    product_names: list[str],
) -> tuple[dict[date, dict[str, int]], dict[date, dict[str, int]]]:
    """期間合計の商品販売数を、日別店舗純売上の比率で按分。"""
    days_sorted = sorted(summary_by_day.keys())
    weights = {d: float(max(0, summary_by_day[d].total_sales)) for d in days_sorted}
    if sum(weights.values()) <= 0:
        weights = {d: 1.0 for d in days_sorted}

    units_by_day: dict[date, dict[str, int]] = {}
    sales_by_day: dict[date, dict[str, int]] = {}
    for name in product_names:
        unit_alloc = allocate_integer_by_weights(period_units.get(name, 0), weights)
        sales_alloc = allocate_integer_by_weights(period_sales.get(name, 0), weights)
        for day_val in days_sorted:
            units_by_day.setdefault(day_val, {n: 0 for n in product_names})
            sales_by_day.setdefault(day_val, {n: 0 for n in product_names})
            units_by_day[day_val][name] = unit_alloc.get(day_val, 0)
            sales_by_day[day_val][name] = sales_alloc.get(day_val, 0)
    return units_by_day, sales_by_day


def list_square_labels_from_matrix(df: pd.DataFrame) -> list[str]:
    """商品別マトリックスCSVに含まれる照合キー一覧（重複なし）。"""
    prepared = prepare_square_dataframe(df)
    if not is_matrix_format_df(prepared):
        return []
    cols = [str(c) for c in prepared.columns]
    item_col = find_matrix_item_column(cols)
    var_col = find_variation_column(cols)
    if not item_col:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for _, row in prepared.iterrows():
        item = str(row[item_col]).strip()
        if not item or is_summary_row_label(item):
            continue
        variation = str(row[var_col]).strip() if var_col else ""
        label = build_square_product_label(item, variation)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return sorted(labels)


def summarize_square_row_mapping(df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """CSV各行がどの登録商品に紐づくか一覧（取り込み前の確認用）。"""
    prepared = prepare_square_dataframe(df)
    if is_product_period_summary_df(prepared):
        product_names = products_df["name"].astype(str).tolist()
        cols = [str(c) for c in prepared.columns]
        item_col = find_column(cols, COLUMN_ALIASES["item"])
        var_col = find_variation_column(cols)
        qty_col = find_column(cols, ["販売数", "販売商品数"] + COLUMN_ALIASES["quantity"])
        rows_out: list[dict[str, Any]] = []
        for _, row in prepared.iterrows():
            item = str(row[item_col]).strip()
            if not item or is_summary_row_label(item):
                continue
            variation = str(row[var_col]).strip() if var_col else ""
            label = build_square_product_label(item, variation)
            matched = match_product_name(label, product_names)
            qty_val = int(pd.to_numeric(row[qty_col], errors="coerce") or 0) if qty_col else 0
            if qty_val == 0:
                continue
            rows_out.append(
                {
                    "Square商品名": item,
                    "バリエーション": variation if variation and variation.lower() not in SKIP_VARIATION_VALUES else "—",
                    "照合キー": label,
                    "アプリ商品": matched or "（未紐づけ）",
                    "紐づけ": "自動" if matched else "—",
                    "期間合計": qty_val,
                }
            )
        return pd.DataFrame(rows_out)
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
        via_custom = bool(matched and resolve_custom_mapping(label, product_names))
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
                "紐づけ": "手動設定" if via_custom else ("自動" if matched else "—"),
                "期間合計": period_total,
            }
        )
    return pd.DataFrame(rows_out)


def parse_matrix_units_by_day(
    df: pd.DataFrame, products_df: pd.DataFrame
) -> tuple[dict[date, dict[str, int]], dict[date, dict[str, int]], dict[date, float], list[str]]:
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
    product_sales_by_day: dict[date, dict[str, int]] = {}
    revenue_by_day: dict[date, float] = {}
    used_yen = False
    unmapped_labels: set[str] = set()

    for day_val, day_group in long_df.groupby("date"):
        units = {name: 0 for name in product_names}
        sales = {name: 0 for name in product_names}
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
                line_sales = raw
                day_revenue += raw
                add_units = max(0, round(raw / unit_price)) if unit_price > 0 else 0
            else:
                add_units = raw
                line_sales = raw * unit_price
                day_revenue += line_sales
            units[matched] = units.get(matched, 0) + add_units
            sales[matched] = sales.get(matched, 0) + int(line_sales)
        if sum(units.values()) > 0 or day_revenue > 0:
            units_by_day[day_val] = units
            product_sales_by_day[day_val] = sales
            revenue_by_day[day_val] = max(day_revenue, 1.0)

    if used_yen:
        warnings.append("セルが金額（¥）表記のため、登録単価から販売数を換算しました。")
    if unmapped_labels:
        sample = "、".join(sorted(unmapped_labels)[:6])
        warnings.append(f"未紐づけのSquare行: {sample}（他{max(0, len(unmapped_labels) - 6)}件）")
    warnings.append(f"商品別CSVから有効な営業日: {len(units_by_day)}日（販売ゼロのみの列は除外）")
    return units_by_day, product_sales_by_day, revenue_by_day, warnings


def parse_matrix_store_gross_by_day(df: pd.DataFrame) -> dict[date, float]:
    """商品別マトリックスの全商品・全行の¥セル合計（店舗の日次総売上ベース）。"""
    prepared = prepare_square_dataframe(df)
    if not is_matrix_format_df(prepared):
        return {}
    long_df = matrix_to_long_dataframe(prepared)
    gross_by_day: dict[date, float] = {}
    for day_val, day_group in long_df.groupby("date"):
        day_total = 0.0
        for _, row in day_group.iterrows():
            if bool(row.get("is_revenue", False)):
                day_total += int(row["quantity"])
        if day_total > 0:
            gross_by_day[day_val] = day_total
    return gross_by_day


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
    sales_col = pick_preferred_sales_column(cols)
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


def parse_sales_summary_matrix(df: pd.DataFrame, *, net_sales_row_only: bool = False) -> tuple[dict[date, DaySummary], list[str]]:
    prepared = prepare_square_dataframe(df)
    cols = [str(c) for c in prepared.columns]
    item_col = resolve_summary_label_column(cols)
    date_cols = get_matrix_date_columns(cols)
    if not item_col or not date_cols:
        raise ValueError("売上サマリーCSVの列構成を認識できませんでした。")

    if net_sales_row_only:
        warnings = ["売上サマリー（日次）: 各行の「純売上高」をそのまま店舗純売上として取り込みました。"]
    else:
        warnings = ["売上サマリーCSV（日別・マトリックス）を読み込みました。"]

    by_day: dict[date, DaySummary] = {}
    for dcol in date_cols:
        if day_val := parse_column_as_date(dcol):
            by_day[day_val] = DaySummary(0, 0)

    sales_by_day: dict[date, tuple[int, int]] = {}
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
            elif net_sales_row_only:
                if is_net_sales_metric_label(label):
                    by_day[day_val] = DaySummary(val, current.total_customers)
            else:
                score = sales_metric_priority_score(label)
                if score >= 0:
                    prev_score, _ = sales_by_day.get(day_val, (-1, 0))
                    if score > prev_score:
                        sales_by_day[day_val] = (score, val)

    if not net_sales_row_only:
        for day_val, (score, val) in sales_by_day.items():
            if score >= 0:
                current = by_day[day_val]
                by_day[day_val] = DaySummary(val, current.total_customers)

    by_day = {d: s for d, s in by_day.items() if s.total_sales > 0 or s.total_customers > 0}
    if not by_day:
        raise ValueError("売上サマリーCSVから日別の純売上高・取引件数を抽出できませんでした。")
    return by_day, warnings


def extract_period_summary_from_kv(df: pd.DataFrame) -> tuple[int, int, int]:
    df = normalize_square_kv_dataframe(df)
    labels, values = kv_metric_series(df)
    metric_map = dict(zip(labels, values))
    customers = 0
    for key in KV_CUSTOMER_METRIC_LABELS:
        if key in metric_map:
            customers = parse_count_metric(metric_map[key])
            break
    net_sales = pick_net_sales_from_kv(metric_map)
    gross_sales = pick_gross_sales_from_kv(metric_map)
    if customers == 0 and net_sales == 0:
        raise ValueError("売上サマリーCSVから期間合計を抽出できませんでした。")
    return customers, net_sales, gross_sales


def weights_are_uniform_calendar(daily_weights: dict[date, float]) -> bool:
    """商品別マトリックスが無いときの均等配分用ウェイト（全日 1.0）。"""
    if not daily_weights:
        return True
    vals = [float(v) for v in daily_weights.values()]
    return len(set(vals)) <= 1 and vals[0] == 1.0


def distribute_period_summary_to_days(
    period_customers: int,
    period_net_sales: int,
    daily_weights: dict[date, float],
    *,
    period_gross_sales: int = 0,
) -> dict[date, DaySummary]:
    """期間の純売上高を日別に配分。総売上高と日次総売上があれば比率で按分。"""
    if not daily_weights:
        raise ValueError("商品別CSVの日付が無いため、期間サマリーを日別に配分できません。")
    days_sorted = sorted(daily_weights.keys())
    by_day: dict[date, DaySummary] = {}
    allocated_sales = allocated_customers = 0

    if period_gross_sales > 0:
        for i, day_val in enumerate(days_sorted):
            day_gross = daily_weights.get(day_val, 0.0)
            if i == len(days_sorted) - 1:
                day_sales = period_net_sales - allocated_sales
                day_customers = period_customers - allocated_customers
            else:
                day_sales = int(period_net_sales * day_gross / period_gross_sales)
                day_customers = int(period_customers * day_gross / period_gross_sales)
                allocated_sales += day_sales
                allocated_customers += day_customers
            by_day[day_val] = DaySummary(max(0, day_sales), max(0, day_customers))
        return by_day

    total_w = sum(daily_weights.values()) or 1.0
    for i, day_val in enumerate(days_sorted):
        w = daily_weights[day_val]
        if i == len(days_sorted) - 1:
            day_sales = period_net_sales - allocated_sales
            day_customers = period_customers - allocated_customers
        else:
            ratio = w / total_w
            day_sales = int(period_net_sales * ratio)
            day_customers = int(period_customers * ratio)
            allocated_sales += day_sales
            allocated_customers += day_customers
        by_day[day_val] = DaySummary(max(0, day_sales), max(0, day_customers))
    return by_day


def parse_sales_summary_csv(
    df: pd.DataFrame, daily_weights: dict[date, float] | None = None
) -> tuple[dict[date, DaySummary], list[str]]:
    prepared = prepare_square_dataframe(df)
    if is_sales_summary_daily_matrix_df(prepared):
        return parse_sales_summary_matrix(prepared, net_sales_row_only=True)

    kv_df = normalize_square_kv_dataframe(prepared if prepared.shape[1] >= 2 else df)
    if is_square_kv_summary_df(kv_df):
        if not daily_weights:
            raise ValueError("期間集計の売上サマリーは、商品別CSVと一緒にアップロードしてください。")
        period_c, period_net, period_gross = extract_period_summary_from_kv(kv_df)
        use_gross_ratio = period_gross > 0 and not weights_are_uniform_calendar(daily_weights)
        warnings = [
            f"売上サマリー（期間集計）: 純売上高 ¥{period_net:,} を日別に配分しました。",
        ]
        if use_gross_ratio:
            warnings.append(
                f"日次配分は商品別CSVの全商品売上（総売上高 ¥{period_gross:,} との比率）で按分しています。"
            )
        else:
            warnings.append(
                "日次配分はファイル名の期間日数で均等に按分しています（日別マトリックスが無いため）。"
            )
        return (
            distribute_period_summary_to_days(
                period_c,
                period_net,
                daily_weights,
                period_gross_sales=period_gross if use_gross_ratio else 0,
            ),
            warnings,
        )

    if is_matrix_format_df(prepared):
        item_col = resolve_summary_label_column([str(c) for c in prepared.columns])
        if item_col and any(
            is_customer_metric_label(str(row[item_col])) or is_sales_metric_label(str(row[item_col]))
            for _, row in prepared.iterrows()
        ):
            return parse_sales_summary_matrix(prepared)
    return parse_sales_summary_vertical(prepared)


def classify_uploaded_csv(df: pd.DataFrame, products_df: pd.DataFrame, filename: str = "") -> str:
    name_lower = (filename or "").lower()
    prepared = prepare_square_dataframe(df)
    if "sales-summary" in name_lower or (
        "売上サマリー" in (filename or "") and "商品" not in (filename or "")
    ):
        return "sales_summary"
    if is_square_kv_summary_df(normalize_square_kv_dataframe(df)):
        return "sales_summary"
    product_names = products_df["name"].astype(str).tolist()
    if "商品売上" in (filename or "") or "商品売上サマリー" in (filename or ""):
        if is_matrix_format_df(prepared):
            return "product_matrix"
        if is_product_period_summary_df(prepared):
            return "product_period_summary"
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
    product_sales_by_day: dict[date, dict[str, int]] | None = None,
) -> tuple[list[DayImport], list[str]]:
    warnings = [
        "ルートB: 2枚のCSVを日付で結合しました。",
        "店舗総売上は税抜金額として取り込みます（純売上高を優先）。",
    ]
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
    skipped = 0
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
        day_summary = DaySummary(total_sales, total_customers)
        if not is_meaningful_day(units, day_summary):
            skipped += 1
            continue
        sales = (product_sales_by_day or {}).get(day_val, {name: 0 for name in product_names})
        imports.append(DayImport(day_val, total_sales, total_customers, units, sales))
    if skipped:
        warnings.append(
            f"販売ゼロかつ売上・客数が極小の日 {skipped}日は除外しました（CSVに無い幽霊日の防止）。"
        )
    if not imports:
        raise ValueError("取り込み可能な有効日がありませんでした。CSVの期間と列をご確認ください。")
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
            units, sales_by_day, _weights, w = parse_matrix_units_by_day(df, products_df)
            product_names = products_df["name"].astype(str).tolist()
            price_map = dict(zip(products_df["name"], products_df["unit_price"]))
            imports = [
                DayImport(
                    d,
                    int(sum(sales_by_day.get(d, {}).values()))
                    or int(sum(units[d].get(n, 0) * int(price_map.get(n, 0)) for n in product_names)),
                    max(100, int(sum(units[d].values()) / 0.12)) if sum(units[d].values()) > 0 else 0,
                    units[d],
                    sales_by_day.get(d, {}),
                )
                for d in sorted(units)
            ]
            return imports, w + ["商品別CSVのみ。客数・店舗売上は推定値です。"]
        if kind == "sales_summary":
            raise ValueError("売上サマリー1枚のみの場合は、商品別マトリックスCSVも必要です。")
        raise ValueError(f"CSV形式を認識できませんでした: {classified[0][0]}")
    product_file = next((x for x in classified if x[2] in ("product_matrix", "product_period_summary")), None)
    summary_file = next((x for x in classified if x[2] == "sales_summary"), None)
    if not product_file or not summary_file:
        kinds = ", ".join(f"{x[0]}→{x[2]}" for x in classified)
        raise ValueError(f"2枚の内訳を認識できませんでした。（判定: {kinds}）")

    if product_file[2] == "product_period_summary":
        period_units, period_sales, w1 = parse_product_period_summary(product_file[1], products_df)
        date_range = parse_date_range_from_filenames(product_file[0], summary_file[0])
        if not date_range:
            raise ValueError(
                "期間集計CSVのファイル名に日付範囲が含まれていません。"
                " 例: 商品売上サマリー-2026-05-01-2026-06-01.csv"
            )
        calendar_weights = {d: 1.0 for d in days_in_range(date_range[0], date_range[1])}
        summary_by_day, w2 = parse_sales_summary_csv(summary_file[1], daily_weights=calendar_weights)
        product_names = products_df["name"].astype(str).tolist()
        units_by_day, sales_by_day = distribute_period_products_to_days(
            period_units, period_sales, summary_by_day, product_names
        )
        w1.append(
            f"期間 {date_range[0]}〜{date_range[1]} の販売数を、売上サマリーの日別純売上比率で按分しました。"
        )
        imports, w3 = merge_dual_csv_imports(units_by_day, summary_by_day, products_df, sales_by_day)
        return imports, w1 + w2 + w3

    units_by_day, sales_by_day, _food_revenue, w1 = parse_matrix_units_by_day(product_file[1], products_df)
    store_gross_by_day = parse_matrix_store_gross_by_day(product_file[1])
    weights = store_gross_by_day or _food_revenue
    summary_by_day, w2 = parse_sales_summary_csv(summary_file[1], daily_weights=weights)
    imports, w3 = merge_dual_csv_imports(units_by_day, summary_by_day, products_df, sales_by_day)
    return imports, w1 + w2 + w3
