"""在庫管理: 予測消費と縦型カレンダー用の投影ロジック。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import shared_storage as store

DATA_DIR = store.DATA_DIR
INVENTORY_CSV = DATA_DIR / "inventory.csv"
DELIVERIES_CSV = DATA_DIR / "inventory_deliveries.csv"
CONSUMPTION_PLAN_CSV = DATA_DIR / "inventory_consumption_plan.csv"
DELIVERY_PLAN_CSV = DATA_DIR / "inventory_delivery_plan.csv"
ACTUAL_SALES_CSV = DATA_DIR / "inventory_actual_sales.csv"

DEFAULT_MAX_STOCK = 300

INVENTORY_COLUMNS = [
    "product_id",
    "product_name",
    "current_stock",
    "safety_stock",
    "max_stock",
    "delivery_weekdays",
    "delivery_quantity",
    "delivery_by_weekday",
    "delivery_by_weekday_parity",
    "updated_at",
]
DELIVERY_COLUMNS = ["delivery_id", "product_id", "delivery_date", "quantity", "memo", "created_at"]
CONSUMPTION_PLAN_COLUMNS = ["product_id", "plan_date", "planned_use", "updated_at"]
DELIVERY_PLAN_COLUMNS = ["product_id", "plan_date", "delivery_qty", "updated_at"]
ACTUAL_SALES_COLUMNS = ["product_id", "sale_date", "actual_units", "updated_at"]

WEEKDAY_LABELS_JA = ["月", "火", "水", "木", "金", "土", "日"]


def to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def init_inventory_csv(products_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if INVENTORY_CSV.exists() and INVENTORY_CSV.stat().st_size > 0:
        sync_inventory_products(products_df)
        return
    now = datetime.now().isoformat()
    rows = [
        {
            "product_id": str(row["product_id"]),
            "product_name": str(row["name"]),
            "current_stock": 0,
            "safety_stock": 10,
            "max_stock": DEFAULT_MAX_STOCK,
            "delivery_weekdays": "",
            "delivery_quantity": 0,
            "delivery_by_weekday": "",
            "delivery_by_weekday_parity": "",
            "updated_at": now,
        }
        for _, row in products_df.iterrows()
        if int(row.get("is_active", 1)) == 1
    ]
    store.write_csv(pd.DataFrame(rows, columns=INVENTORY_COLUMNS), INVENTORY_CSV)


def init_deliveries_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DELIVERIES_CSV.exists() or DELIVERIES_CSV.stat().st_size == 0:
        store.write_csv(pd.DataFrame(columns=DELIVERY_COLUMNS), DELIVERIES_CSV)


def init_consumption_plan_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONSUMPTION_PLAN_CSV.exists() or CONSUMPTION_PLAN_CSV.stat().st_size == 0:
        store.write_csv(pd.DataFrame(columns=CONSUMPTION_PLAN_COLUMNS), CONSUMPTION_PLAN_CSV)


def load_consumption_plan_df(product_id: str | None = None) -> pd.DataFrame:
    if not CONSUMPTION_PLAN_CSV.exists() or CONSUMPTION_PLAN_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=CONSUMPTION_PLAN_COLUMNS)
    df = store.read_csv(CONSUMPTION_PLAN_CSV)
    df["planned_use"] = pd.to_numeric(df["planned_use"], errors="coerce").fillna(0).astype(int)
    df["plan_date"] = pd.to_datetime(df["plan_date"], errors="coerce").dt.date
    if product_id:
        df = df[df["product_id"].astype(str) == str(product_id)]
    return df


def load_manual_planned_use(
    product_id: str, start_date: date, end_date: date
) -> dict[date, int]:
    df = load_consumption_plan_df(product_id)
    if df.empty:
        return {}
    out: dict[date, int] = {}
    for _, row in df.iterrows():
        d = row["plan_date"]
        if d is None or pd.isna(d):
            continue
        if isinstance(d, pd.Timestamp):
            d = d.date()
        if start_date <= d <= end_date:
            out[d] = int(row["planned_use"])
    return out


def save_consumption_plan(product_id: str, planned_by_day: dict[date, int]) -> bool:
    """指定期間の手入力計画消費を保存。"""
    init_consumption_plan_csv()
    df = load_consumption_plan_df()
    pid = str(product_id)
    if not df.empty:
        drop_days = set(planned_by_day.keys())
        df = df[~((df["product_id"].astype(str) == pid) & (df["plan_date"].isin(drop_days)))]
    now = datetime.now().isoformat()
    new_rows: list[dict[str, Any]] = []
    for day_val, qty in planned_by_day.items():
        if qty < 0:
            continue
        new_rows.append(
            {
                "product_id": pid,
                "plan_date": day_val.isoformat(),
                "planned_use": int(qty),
                "updated_at": now,
            }
        )
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    if df.empty:
        init_consumption_plan_csv()
        return True
    return store.write_csv(df, CONSUMPTION_PLAN_CSV)


def clear_consumption_plan(product_id: str, start_date: date, end_date: date) -> bool:
    df = load_consumption_plan_df()
    if df.empty:
        return True
    pid = str(product_id)
    mask = (df["product_id"].astype(str) == pid) & (df["plan_date"] >= start_date) & (df["plan_date"] <= end_date)
    df = df[~mask]
    if df.empty:
        init_consumption_plan_csv()
        return True
    return store.write_csv(df, CONSUMPTION_PLAN_CSV)


def init_actual_sales_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ACTUAL_SALES_CSV.exists() or ACTUAL_SALES_CSV.stat().st_size == 0:
        store.write_csv(pd.DataFrame(columns=ACTUAL_SALES_COLUMNS), ACTUAL_SALES_CSV)


def load_actual_sales_df(product_id: str | None = None) -> pd.DataFrame:
    if not ACTUAL_SALES_CSV.exists() or ACTUAL_SALES_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=ACTUAL_SALES_COLUMNS)
    df = store.read_csv(ACTUAL_SALES_CSV)
    df["actual_units"] = pd.to_numeric(df["actual_units"], errors="coerce").fillna(0).astype(int)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date
    if product_id:
        df = df[df["product_id"].astype(str) == str(product_id)]
    return df


def load_manual_actual_sales(
    product_id: str, start_date: date, end_date: date
) -> dict[date, int]:
    df = load_actual_sales_df(product_id)
    if df.empty:
        return {}
    out: dict[date, int] = {}
    for _, row in df.iterrows():
        d = row["sale_date"]
        if d is None or pd.isna(d):
            continue
        if isinstance(d, pd.Timestamp):
            d = d.date()
        if start_date <= d <= end_date:
            out[d] = int(row["actual_units"])
    return out


def save_actual_sales(product_id: str, actual_by_day: dict[date, int]) -> bool:
    """手入力の実売数を保存。"""
    init_actual_sales_csv()
    df = load_actual_sales_df()
    pid = str(product_id)
    if not df.empty:
        drop_days = set(actual_by_day.keys())
        df = df[~((df["product_id"].astype(str) == pid) & (df["sale_date"].isin(drop_days)))]
    now = datetime.now().isoformat()
    new_rows: list[dict[str, Any]] = []
    for day_val, qty in actual_by_day.items():
        if qty < 0:
            continue
        new_rows.append(
            {
                "product_id": pid,
                "sale_date": day_val.isoformat(),
                "actual_units": int(qty),
                "updated_at": now,
            }
        )
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    if df.empty:
        init_actual_sales_csv()
        return True
    return store.write_csv(df, ACTUAL_SALES_CSV)


def clear_actual_sales(product_id: str, start_date: date, end_date: date) -> bool:
    df = load_actual_sales_df()
    if df.empty:
        return True
    pid = str(product_id)
    mask = (
        (df["product_id"].astype(str) == pid)
        & (df["sale_date"] >= start_date)
        & (df["sale_date"] <= end_date)
    )
    df = df[~mask]
    if df.empty:
        init_actual_sales_csv()
        return True
    return store.write_csv(df, ACTUAL_SALES_CSV)


def replace_actual_sales_for_period(
    product_id: str,
    start_date: date,
    end_date: date,
    actual_by_day: dict[date, int],
    *,
    today: date,
) -> bool:
    """表示期間のうち今日以前の実売数を置き換え（未入力日は削除）。"""
    past_end = min(end_date, today)
    if past_end < start_date:
        return True
    ok = clear_actual_sales(product_id, start_date, past_end)
    filtered = {d: int(v) for d, v in actual_by_day.items() if start_date <= d <= past_end}
    if filtered:
        ok = save_actual_sales(product_id, filtered) and ok
    return ok


def init_delivery_plan_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DELIVERY_PLAN_CSV.exists() or DELIVERY_PLAN_CSV.stat().st_size == 0:
        store.write_csv(pd.DataFrame(columns=DELIVERY_PLAN_COLUMNS), DELIVERY_PLAN_CSV)


def load_delivery_plan_df(product_id: str | None = None) -> pd.DataFrame:
    if not DELIVERY_PLAN_CSV.exists() or DELIVERY_PLAN_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=DELIVERY_PLAN_COLUMNS)
    df = store.read_csv(DELIVERY_PLAN_CSV)
    df["delivery_qty"] = pd.to_numeric(df["delivery_qty"], errors="coerce").fillna(0).astype(int)
    df["plan_date"] = pd.to_datetime(df["plan_date"], errors="coerce").dt.date
    if product_id:
        df = df[df["product_id"].astype(str) == str(product_id)]
    return df


def load_manual_delivery_plan(
    product_id: str, start_date: date, end_date: date
) -> dict[date, int]:
    df = load_delivery_plan_df(product_id)
    if df.empty:
        return {}
    out: dict[date, int] = {}
    for _, row in df.iterrows():
        d = row["plan_date"]
        if d is None or pd.isna(d):
            continue
        if isinstance(d, pd.Timestamp):
            d = d.date()
        if start_date <= d <= end_date:
            out[d] = int(row["delivery_qty"])
    return out


def save_delivery_plan(product_id: str, delivery_by_day: dict[date, int]) -> bool:
    """カレンダーで入力した日別納品数を保存（0は行削除）。"""
    init_delivery_plan_csv()
    df = load_delivery_plan_df()
    pid = str(product_id)
    if not df.empty:
        drop_days = set(delivery_by_day.keys())
        df = df[~((df["product_id"].astype(str) == pid) & (df["plan_date"].isin(drop_days)))]
    now = datetime.now().isoformat()
    new_rows: list[dict[str, Any]] = []
    for day_val, qty in delivery_by_day.items():
        if int(qty) <= 0:
            continue
        new_rows.append(
            {
                "product_id": pid,
                "plan_date": day_val.isoformat(),
                "delivery_qty": int(qty),
                "updated_at": now,
            }
        )
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    if df.empty:
        init_delivery_plan_csv()
        return True
    return store.write_csv(df, DELIVERY_PLAN_CSV)


def clear_delivery_plan(product_id: str, start_date: date, end_date: date) -> bool:
    df = load_delivery_plan_df()
    if df.empty:
        return True
    pid = str(product_id)
    plan_dates = df["plan_date"].apply(
        lambda d: d.date() if isinstance(d, pd.Timestamp) else d
    )
    mask = (
        (df["product_id"].astype(str) == pid)
        & (plan_dates >= start_date)
        & (plan_dates <= end_date)
    )
    df = df[~mask]
    if df.empty:
        init_delivery_plan_csv()
        return True
    return store.write_csv(df, DELIVERY_PLAN_CSV)


def replace_delivery_plan_for_period(
    product_id: str,
    start_date: date,
    end_date: date,
    delivery_by_day: dict[date, int],
) -> bool:
    """表示期間の納品数をカレンダー表の内容で置き換え（0は削除）。"""
    ok = clear_delivery_plan(product_id, start_date, end_date)
    positive = {d: int(v) for d, v in delivery_by_day.items() if int(v) > 0}
    if positive:
        ok = save_delivery_plan(product_id, positive) and ok
    return ok


def get_delivery_weekdays(product_id: str) -> list[int]:
    """納品がある曜日（0=月）。数量はカレンダーで入力。"""
    df = load_inventory_df()
    row = df[df["product_id"].astype(str) == str(product_id)]
    if row.empty:
        return []
    wds = parse_weekday_list(str(row["delivery_weekdays"].iloc[0]))
    if wds:
        return wds
    parity = parse_delivery_by_weekday_parity(str(row.get("delivery_by_weekday_parity", "").iloc[0]))
    if parity:
        return sorted(parity.keys())
    base = parse_delivery_by_weekday(str(row.get("delivery_by_weekday", "").iloc[0]))
    return sorted(base.keys()) if base else []


def save_delivery_weekdays(product_id: str, weekdays: list[int]) -> None:
    """納品曜日のみ保存（数量はカレンダー表で入力）。"""
    cleaned = sorted({int(w) for w in weekdays if 0 <= int(w) <= 6})
    df = load_inventory_df()
    pid = str(product_id)
    now = datetime.now().isoformat()
    mask = df["product_id"].astype(str) == pid
    payload = {
        "delivery_weekdays": format_weekday_list(cleaned),
        "delivery_quantity": 0,
        "delivery_by_weekday": "",
        "delivery_by_weekday_parity": "",
        "updated_at": now,
    }
    if mask.any():
        for key, val in payload.items():
            df.loc[mask, key] = val
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "product_id": pid,
                            "product_name": "",
                            "current_stock": 0,
                            "safety_stock": 10,
                            "max_stock": DEFAULT_MAX_STOCK,
                            **payload,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    store.write_csv(df, INVENTORY_CSV)


def sync_inventory_products(products_df: pd.DataFrame) -> None:
    """販売中商品を在庫マスタに反映（新規追加・商品名の更新）。"""
    if not INVENTORY_CSV.exists() or INVENTORY_CSV.stat().st_size == 0:
        init_inventory_csv(products_df)
        return
    inv = store.read_csv(INVENTORY_CSV)
    existing = set(inv["product_id"].astype(str))
    now = datetime.now().isoformat()
    new_rows: list[dict[str, Any]] = []
    changed = False
    for _, row in products_df.iterrows():
        if int(row.get("is_active", 1)) != 1:
            continue
        pid = str(row["product_id"])
        pname = str(row["name"])
        if pid not in existing:
            new_rows.append(
                {
                    "product_id": pid,
                    "product_name": pname,
                    "current_stock": 0,
                    "safety_stock": 10,
                    "max_stock": DEFAULT_MAX_STOCK,
                    "delivery_weekdays": "",
                    "delivery_quantity": 0,
                    "delivery_by_weekday": "",
                    "delivery_by_weekday_parity": "",
                    "updated_at": now,
                }
            )
        else:
            mask = inv["product_id"].astype(str) == pid
            if mask.any() and str(inv.loc[mask, "product_name"].iloc[0]) != pname:
                inv.loc[mask, "product_name"] = pname
                inv.loc[mask, "updated_at"] = now
                changed = True
    if new_rows:
        inv = pd.concat([inv, pd.DataFrame(new_rows)], ignore_index=True)
        changed = True
    if changed:
        store.write_csv(inv, INVENTORY_CSV)


def _coerce_inventory_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSVの空欄が float/NaN になる列を文字列に揃える（'' 代入エラー防止）。"""
    for col in ("delivery_weekdays", "delivery_by_weekday", "delivery_by_weekday_parity", "updated_at"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str).replace("nan", "", regex=False)
    return df


def load_inventory_df() -> pd.DataFrame:
    if not INVENTORY_CSV.exists() or INVENTORY_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    df = store.read_csv(INVENTORY_CSV, dtype=str, keep_default_na=False)
    for col in INVENTORY_COLUMNS:
        if col not in df.columns:
            if col in ("delivery_weekdays", "delivery_by_weekday", "delivery_by_weekday_parity"):
                df[col] = ""
            elif col == "max_stock":
                df[col] = str(DEFAULT_MAX_STOCK)
            else:
                df[col] = "0" if col != "updated_at" else ""
    df = _coerce_inventory_text_columns(df)
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0).astype(int)
    df["safety_stock"] = pd.to_numeric(df["safety_stock"], errors="coerce").fillna(0).astype(int)
    df["max_stock"] = pd.to_numeric(df.get("max_stock", DEFAULT_MAX_STOCK), errors="coerce").fillna(
        DEFAULT_MAX_STOCK
    ).astype(int)
    df.loc[df["max_stock"] <= 0, "max_stock"] = DEFAULT_MAX_STOCK
    df["delivery_quantity"] = pd.to_numeric(df["delivery_quantity"], errors="coerce").fillna(0).astype(int)
    return df


def parse_weekday_list(text: str) -> list[int]:
    if not str(text).strip():
        return []
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            w = int(part)
            if 0 <= w <= 6:
                out.append(w)
        except ValueError:
            continue
    return sorted(set(out))


def format_weekday_list(weekdays: list[int]) -> str:
    return ",".join(str(w) for w in sorted(set(weekdays)))


def weekday_labels_text(weekdays: list[int]) -> str:
    if not weekdays:
        return "未設定"
    return "・".join(WEEKDAY_LABELS_JA[w] for w in sorted(weekdays))


def parse_delivery_by_weekday(text: str) -> dict[int, int]:
    raw = str(text).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        out: dict[int, int] = {}
        for key, val in data.items():
            w = int(key)
            q = int(val)
            if 0 <= w <= 6 and q > 0:
                out[w] = q
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def encode_delivery_by_weekday(schedule: dict[int, int]) -> str:
    filtered = {str(w): int(q) for w, q in schedule.items() if 0 <= w <= 6 and int(q) > 0}
    return json.dumps(filtered, ensure_ascii=False) if filtered else ""


def weekday_schedule_summary(schedule: dict[int, int]) -> str:
    if not schedule:
        return "未設定"
    return " · ".join(f"{WEEKDAY_LABELS_JA[w]}{q}個" for w, q in sorted(schedule.items()))


def get_weekday_delivery_schedule(product_id: str) -> dict[int, int]:
    """曜日ごとの定期納品数。キー=weekday(0=月)。"""
    df = load_inventory_df()
    row = df[df["product_id"].astype(str) == str(product_id)]
    if row.empty:
        return {}
    by_wd = parse_delivery_by_weekday(str(row.get("delivery_by_weekday", "").iloc[0]))
    if by_wd:
        return by_wd
    weekdays = parse_weekday_list(str(row["delivery_weekdays"].iloc[0]))
    qty = int(row["delivery_quantity"].iloc[0])
    if weekdays and qty > 0:
        return {w: qty for w in weekdays}
    return {}


def save_weekday_delivery_schedule(product_id: str, schedule: dict[int, int]) -> None:
    """曜日ごとの納品数を保存。"""
    cleaned = {int(w): int(q) for w, q in schedule.items() if 0 <= int(w) <= 6 and int(q) > 0}
    df = load_inventory_df()
    pid = str(product_id)
    now = datetime.now().isoformat()
    mask = df["product_id"].astype(str) == pid
    payload = {
        "delivery_by_weekday": encode_delivery_by_weekday(cleaned),
        "delivery_weekdays": format_weekday_list(list(cleaned.keys())),
        "delivery_quantity": max(cleaned.values()) if cleaned else 0,
        "updated_at": now,
    }
    if mask.any():
        for key, val in payload.items():
            df.loc[mask, key] = val
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "product_id": pid,
                            "product_name": "",
                            "current_stock": 0,
                            "safety_stock": 10,
                            "max_stock": DEFAULT_MAX_STOCK,
                            **payload,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    store.write_csv(df, INVENTORY_CSV)


def week_parity_iso(d: date) -> int:
    """ISO週番号の奇数/偶数（1=奇数週, 0=偶数週）。"""
    return int(d.isocalendar().week % 2)


def parse_delivery_by_weekday_parity(text: str) -> dict[int, dict[int, int]]:
    """
    例:
      {"0":{"0":30,"1":40},"3":{"0":0,"1":50}}
    - weekday: 0=月 ... 6=日
    - parity: 0=偶数週, 1=奇数週
    """
    raw = str(text).strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        out: dict[int, dict[int, int]] = {}
        for wd_key, parity_map in data.items():
            wd = int(wd_key)
            if not (0 <= wd <= 6):
                continue
            if not isinstance(parity_map, dict):
                continue
            even = int(parity_map.get("0", 0) or 0)
            odd = int(parity_map.get("1", 0) or 0)
            if even > 0 or odd > 0:
                out[wd] = {0: max(0, even), 1: max(0, odd)}
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def encode_delivery_by_weekday_parity(schedule: dict[int, dict[int, int]]) -> str:
    filtered: dict[str, dict[str, int]] = {}
    for wd, parity_map in (schedule or {}).items():
        wd = int(wd)
        if not (0 <= wd <= 6):
            continue
        even = int(parity_map.get(0, 0) or 0)
        odd = int(parity_map.get(1, 0) or 0)
        if even <= 0 and odd <= 0:
            continue
        filtered[str(wd)] = {"0": max(0, even), "1": max(0, odd)}
    return json.dumps(filtered, ensure_ascii=False) if filtered else ""


def weekday_schedule_parity_summary(schedule: dict[int, dict[int, int]]) -> str:
    if not schedule:
        return "未設定"
    parts: list[str] = []
    for wd in sorted(schedule.keys()):
        even_qty = int(schedule[wd].get(0, 0))
        odd_qty = int(schedule[wd].get(1, 0))
        if even_qty > 0 and odd_qty > 0:
            parts.append(f"{WEEKDAY_LABELS_JA[wd]}（偶{even_qty}・奇{odd_qty}）")
        elif even_qty > 0:
            parts.append(f"{WEEKDAY_LABELS_JA[wd]}（偶{even_qty}）")
        elif odd_qty > 0:
            parts.append(f"{WEEKDAY_LABELS_JA[wd]}（奇{odd_qty}）")
    return "・".join(parts)


def get_weekday_delivery_schedule_parity(product_id: str) -> dict[int, dict[int, int]]:
    """
    隔週（偶数週/奇数週）で納品数を変える。
    戻り値: {weekday: {0:even_qty, 1:odd_qty}}
    """
    df = load_inventory_df()
    row = df[df["product_id"].astype(str) == str(product_id)]
    if row.empty:
        return {}
    parsed = parse_delivery_by_weekday_parity(str(row.get("delivery_by_weekday_parity", "").iloc[0]))
    if parsed:
        return parsed
    # 旧データ互換: delivery_by_weekday を偶/奇に同じ値で展開
    base = get_weekday_delivery_schedule(product_id)
    if not base:
        return {}
    return {wd: {0: qty, 1: qty} for wd, qty in base.items() if int(qty) > 0}


def save_weekday_delivery_schedule_parity(
    product_id: str,
    schedule: dict[int, dict[int, int]],
) -> None:
    """
    schedule: {weekday: {0:even_qty, 1:odd_qty}}
    """
    df = load_inventory_df()
    pid = str(product_id)
    now = datetime.now().isoformat()
    mask = df["product_id"].astype(str) == pid

    cleaned: dict[int, dict[int, int]] = {}
    for wd, parity_map in (schedule or {}).items():
        wd = int(wd)
        if not (0 <= wd <= 6):
            continue
        even = int(parity_map.get(0, 0) or 0)
        odd = int(parity_map.get(1, 0) or 0)
        if even <= 0 and odd <= 0:
            continue
        cleaned[wd] = {0: max(0, even), 1: max(0, odd)}

    parity_json = encode_delivery_by_weekday_parity(cleaned)
    # 旧UI/旧互換フィールドも更新しておく（最大値で代表）
    max_map = {wd: max(int(v.get(0, 0)), int(v.get(1, 0))) for wd, v in cleaned.items()}
    delivery_weekdays_str = format_weekday_list(list(max_map.keys()))
    delivery_quantity = max(max_map.values()) if max_map else 0
    delivery_by_weekday_json = encode_delivery_by_weekday(max_map)

    payload = {
        "delivery_by_weekday_parity": parity_json,
        "delivery_weekdays": delivery_weekdays_str,
        "delivery_quantity": delivery_quantity,
        "delivery_by_weekday": delivery_by_weekday_json,
        "updated_at": now,
    }

    if mask.any():
        for key, val in payload.items():
            df.loc[mask, key] = val
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "product_id": pid,
                            "product_name": "",
                            "current_stock": 0,
                            "safety_stock": 10,
                            "max_stock": DEFAULT_MAX_STOCK,
                            **payload,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    store.write_csv(df, INVENTORY_CSV)


def build_combined_delivery_maps(
    product_id: str,
    start_date: date,
    end_date: date,
    one_off_deliveries: pd.DataFrame | None = None,
) -> tuple[dict[date, int], dict[date, int], dict[date, int]]:
    """カレンダー入力（日別）+ 臨時（日付指定）の入荷マップ。total, scheduled, extra。"""
    scheduled = load_manual_delivery_plan(product_id, start_date, end_date)
    extra: dict[date, int] = {}
    if one_off_deliveries is not None and not one_off_deliveries.empty:
        for _, row in one_off_deliveries.iterrows():
            d = row["delivery_date"]
            if isinstance(d, pd.Timestamp):
                d = d.date()
            if d is None or pd.isna(d):
                continue
            if start_date <= d <= end_date:
                extra[d] = extra.get(d, 0) + int(row["quantity"])
    total: dict[date, int] = {}
    for d in set(scheduled) | set(extra):
        total[d] = scheduled.get(d, 0) + extra.get(d, 0)
    return total, scheduled, extra


def save_product_inventory(
    product_id: str,
    product_name: str,
    current_stock: int,
    safety_stock: int,
    max_stock: int | None = None,
) -> None:
    """在庫数・安全在庫・収納MAXを保存。"""
    df = load_inventory_df()
    pid = str(product_id)
    now = datetime.now().isoformat()
    mask = df["product_id"].astype(str) == pid
    max_val = int(max_stock) if max_stock is not None else DEFAULT_MAX_STOCK
    if max_val <= 0:
        max_val = DEFAULT_MAX_STOCK
    if mask.any():
        df.loc[mask, "current_stock"] = int(current_stock)
        df.loc[mask, "safety_stock"] = int(safety_stock)
        df.loc[mask, "max_stock"] = max_val
        df.loc[mask, "product_name"] = product_name
        df.loc[mask, "updated_at"] = now
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "product_id": pid,
                            "product_name": product_name,
                            "current_stock": int(current_stock),
                            "safety_stock": int(safety_stock),
                            "max_stock": max_val,
                            "delivery_weekdays": "",
                            "delivery_quantity": 0,
                            "delivery_by_weekday": "",
                            "delivery_by_weekday_parity": "",
                            "updated_at": now,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    store.write_csv(df, INVENTORY_CSV)


def load_deliveries_df(product_id: str | None = None) -> pd.DataFrame:
    if not DELIVERIES_CSV.exists() or DELIVERIES_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=DELIVERY_COLUMNS)
    df = store.read_csv(DELIVERIES_CSV)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce").dt.date
    if product_id:
        df = df[df["product_id"].astype(str) == str(product_id)]
    return df.sort_values("delivery_date")


def add_delivery(product_id: str, delivery_date: date, quantity: int, memo: str = "") -> None:
    df = load_deliveries_df()
    delivery_id = f"DLV_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(df)}"
    row = pd.DataFrame(
        [
            {
                "delivery_id": delivery_id,
                "product_id": str(product_id),
                "delivery_date": delivery_date.isoformat(),
                "quantity": int(quantity),
                "memo": memo.strip(),
                "created_at": datetime.now().isoformat(),
            }
        ]
    )
    df = pd.concat([df, row], ignore_index=True)
    store.write_csv(df, DELIVERIES_CSV)


def delete_delivery(delivery_id: str) -> None:
    df = load_deliveries_df()
    df = df[df["delivery_id"].astype(str) != str(delivery_id)]
    if df.empty:
        init_deliveries_csv()
    else:
        store.write_csv(df, DELIVERIES_CSV)


def predict_daily_units(
    daily_df: pd.DataFrame,
    product_id: str,
    target_day: date,
    *,
    lookback_days: int = 28,
) -> float:
    """同曜日の過去販売平均を予測消費量とする（データが無ければ期間平均）。"""
    if daily_df.empty:
        return 0.0
    ref_ts = to_ts(target_day)
    mask = (
        (daily_df["product_id"] == product_id)
        & (daily_df["date"].dt.weekday == ref_ts.weekday())
        & (daily_df["date"] < ref_ts)
    )
    hist = daily_df.loc[mask]
    recent = hist[hist["date"] >= to_ts(target_day - timedelta(days=lookback_days))]
    if not recent.empty:
        return float(recent["units_sold"].mean())

    sub = daily_df[
        (daily_df["product_id"] == product_id) & (daily_df["date"] < ref_ts)
    ]
    if len(sub) >= 7:
        sub = sub[sub["date"] >= to_ts(target_day - timedelta(days=lookback_days))]
    if sub.empty:
        return 0.0
    return float(sub["units_sold"].mean())


def build_inventory_projection(
    daily_df: pd.DataFrame,
    product_id: str,
    start_stock: int,
    start_date: date,
    horizon_days: int,
    deliveries: pd.DataFrame | None = None,
    safety_stock: int = 0,
    max_stock: int = 0,
) -> pd.DataFrame:
    """日別の予測消費・入荷・予想在庫を計算（縦型カレンダー用）。"""
    end_date = start_date + timedelta(days=horizon_days - 1)
    delivery_map, scheduled_map, extra_map = build_combined_delivery_maps(
        product_id, start_date, end_date, deliveries
    )

    rows: list[dict[str, Any]] = []
    stock = int(start_stock)
    for offset in range(horizon_days):
        day_val = start_date + timedelta(days=offset)
        predicted = predict_daily_units(daily_df, product_id, day_val)
        use = max(0, int(round(predicted)))
        inbound = int(delivery_map.get(day_val, 0))
        inbound_scheduled = int(scheduled_map.get(day_val, 0))
        inbound_extra = int(extra_map.get(day_val, 0))
        stock_after = stock - use + inbound
        weekday = day_val.weekday()
        rows.append(
            {
                "date": day_val,
                "weekday": weekday,
                "曜日": WEEKDAY_LABELS_JA[weekday],
                "label": f"{WEEKDAY_LABELS_JA[weekday]} {day_val.month}/{day_val.day}",
                "predicted_use": use,
                "planned_use": use,
                "delivery": inbound,
                "delivery_scheduled": inbound_scheduled,
                "delivery_extra": inbound_extra,
                "stock_start": stock,
                "stock_end": stock_after,
                "is_today": offset == 0,
                "is_manual": False,
            }
        )
        stock = stock_after

    out = pd.DataFrame(rows)
    return apply_planned_consumption(out, {}, safety_stock, max_stock)


def _apply_stock_status(
    projection: pd.DataFrame,
    safety_stock: int,
    max_stock: int,
) -> pd.DataFrame:
    """日末在庫から状態ラベルを付与（不足・注意・収納超過）。"""
    out = projection.copy()
    out["status"] = "ok"
    out.loc[out["stock_end"] < 0, "status"] = "out"
    if int(max_stock) > 0:
        out.loc[out["stock_end"] > int(max_stock), "status"] = "over"
    out.loc[
        (out["status"] == "ok") & (out["stock_end"] < int(safety_stock)),
        "status",
    ] = "low"
    return out


def apply_planned_consumption(
    projection: pd.DataFrame,
    manual_by_day: dict[date, int],
    safety_stock: int,
    max_stock: int = 0,
) -> pd.DataFrame:
    """計画消費（手入力含む）で日末在庫・状態を再計算。"""
    if projection.empty:
        return projection
    out = projection.copy()
    stock = int(out.iloc[0]["stock_start"])
    for i in range(len(out)):
        row = out.iloc[i]
        day_val = row["date"]
        if isinstance(day_val, pd.Timestamp):
            day_val = day_val.date()
        predicted = int(row["predicted_use"])
        if day_val in manual_by_day:
            use = max(0, int(manual_by_day[day_val]))
            out.at[i, "is_manual"] = True
        else:
            use = max(0, int(row.get("planned_use", predicted)))
            out.at[i, "is_manual"] = bool(row.get("is_manual", False))
        inbound = int(row["delivery"])
        out.at[i, "planned_use"] = use
        out.at[i, "stock_start"] = stock
        stock_after = stock - use + inbound
        out.at[i, "stock_end"] = stock_after
        stock = stock_after

    return _apply_stock_status(out, safety_stock, max_stock)


def apply_calendar_inputs(
    projection: pd.DataFrame,
    planned_by_day: dict[date, int],
    actual_by_day: dict[date, int],
    *,
    today: date,
    safety_stock: int,
    max_stock: int = 0,
) -> pd.DataFrame:
    """計画消費・実売数（手入力）を反映して日末在庫を再計算。"""
    if projection.empty:
        return projection
    out = projection.copy()
    stock = int(out.iloc[0]["stock_start"])
    for i in range(len(out)):
        row = out.iloc[i]
        day_val = row["date"]
        if isinstance(day_val, pd.Timestamp):
            day_val = day_val.date()
        predicted = int(row["predicted_use"])
        planned = max(0, int(planned_by_day.get(day_val, row.get("planned_use", predicted))))
        is_manual_planned = day_val in planned_by_day
        is_actual = False
        if day_val <= today and day_val in actual_by_day:
            effective = max(0, int(actual_by_day[day_val]))
            is_actual = True
        else:
            effective = planned
        inbound = int(row["delivery"])
        out.at[i, "planned_use"] = planned
        out.at[i, "actual_use"] = int(actual_by_day[day_val]) if day_val in actual_by_day else pd.NA
        out.at[i, "effective_use"] = effective
        out.at[i, "is_manual"] = is_manual_planned
        out.at[i, "is_actual"] = is_actual
        out.at[i, "stock_start"] = stock
        stock_after = stock - effective + inbound
        out.at[i, "stock_end"] = stock_after
        stock = stock_after
    return _apply_stock_status(out, safety_stock, max_stock)


def days_until_stockout(projection: pd.DataFrame) -> int | None:
    """在庫切れまでの日数（当日含む）。切れなければ None。"""
    if projection.empty:
        return None
    for i, row in projection.iterrows():
        if int(row["stock_end"]) < 0:
            return int(i)
    return None
