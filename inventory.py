"""在庫管理: 予測消費と縦型カレンダー用の投影ロジック。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
DELIVERIES_CSV = DATA_DIR / "inventory_deliveries.csv"

INVENTORY_COLUMNS = ["product_id", "product_name", "current_stock", "safety_stock", "updated_at"]
DELIVERY_COLUMNS = ["delivery_id", "product_id", "delivery_date", "quantity", "memo", "created_at"]

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
            "updated_at": now,
        }
        for _, row in products_df.iterrows()
        if int(row.get("is_active", 1)) == 1
    ]
    pd.DataFrame(rows, columns=INVENTORY_COLUMNS).to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")


def init_deliveries_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DELIVERIES_CSV.exists() or DELIVERIES_CSV.stat().st_size == 0:
        pd.DataFrame(columns=DELIVERY_COLUMNS).to_csv(DELIVERIES_CSV, index=False, encoding="utf-8-sig")


def sync_inventory_products(products_df: pd.DataFrame) -> None:
    """新規商品を在庫マスタに追加。"""
    if not INVENTORY_CSV.exists():
        init_inventory_csv(products_df)
        return
    inv = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    existing = set(inv["product_id"].astype(str))
    now = datetime.now().isoformat()
    new_rows: list[dict[str, Any]] = []
    for _, row in products_df.iterrows():
        if int(row.get("is_active", 1)) != 1:
            continue
        pid = str(row["product_id"])
        if pid not in existing:
            new_rows.append(
                {
                    "product_id": pid,
                    "product_name": str(row["name"]),
                    "current_stock": 0,
                    "safety_stock": 10,
                    "updated_at": now,
                }
            )
    if new_rows:
        inv = pd.concat([inv, pd.DataFrame(new_rows)], ignore_index=True)
        inv.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")


def load_inventory_df() -> pd.DataFrame:
    if not INVENTORY_CSV.exists() or INVENTORY_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    df = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(0).astype(int)
    df["safety_stock"] = pd.to_numeric(df["safety_stock"], errors="coerce").fillna(0).astype(int)
    return df


def save_product_inventory(
    product_id: str,
    product_name: str,
    current_stock: int,
    safety_stock: int,
) -> None:
    """在庫数・安全在庫を保存。"""
    df = load_inventory_df()
    pid = str(product_id)
    now = datetime.now().isoformat()
    mask = df["product_id"].astype(str) == pid
    if mask.any():
        df.loc[mask, "current_stock"] = int(current_stock)
        df.loc[mask, "safety_stock"] = int(safety_stock)
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
                            "updated_at": now,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    df.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")


def load_deliveries_df(product_id: str | None = None) -> pd.DataFrame:
    if not DELIVERIES_CSV.exists() or DELIVERIES_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=DELIVERY_COLUMNS)
    df = pd.read_csv(DELIVERIES_CSV, encoding="utf-8-sig")
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
    df.to_csv(DELIVERIES_CSV, index=False, encoding="utf-8-sig")


def delete_delivery(delivery_id: str) -> None:
    df = load_deliveries_df()
    df = df[df["delivery_id"].astype(str) != str(delivery_id)]
    if df.empty:
        init_deliveries_csv()
    else:
        df.to_csv(DELIVERIES_CSV, index=False, encoding="utf-8-sig")


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
) -> pd.DataFrame:
    """日別の予測消費・入荷・予想在庫を計算（縦型カレンダー用）。"""
    delivery_map: dict[date, int] = {}
    if deliveries is not None and not deliveries.empty:
        for _, row in deliveries.iterrows():
            d = row["delivery_date"]
            if isinstance(d, pd.Timestamp):
                d = d.date()
            delivery_map[d] = delivery_map.get(d, 0) + int(row["quantity"])

    rows: list[dict[str, Any]] = []
    stock = int(start_stock)
    for offset in range(horizon_days):
        day_val = start_date + timedelta(days=offset)
        predicted = predict_daily_units(daily_df, product_id, day_val)
        use = max(0, int(round(predicted)))
        inbound = int(delivery_map.get(day_val, 0))
        stock_after = stock - use + inbound
        weekday = day_val.weekday()
        rows.append(
            {
                "date": day_val,
                "weekday": weekday,
                "曜日": WEEKDAY_LABELS_JA[weekday],
                "label": f"{WEEKDAY_LABELS_JA[weekday]} {day_val.month}/{day_val.day}",
                "predicted_use": use,
                "delivery": inbound,
                "stock_start": stock,
                "stock_end": stock_after,
                "is_today": offset == 0,
            }
        )
        stock = stock_after

    out = pd.DataFrame(rows)
    out["status"] = "ok"
    if not out.empty:
        out.loc[out["stock_end"] < 0, "status"] = "out"
        out.loc[(out["stock_end"] >= 0) & (out["stock_end"] < int(safety_stock)), "status"] = "low"
    return out


def days_until_stockout(projection: pd.DataFrame) -> int | None:
    """在庫切れまでの日数（当日含む）。切れなければ None。"""
    if projection.empty:
        return None
    for i, row in projection.iterrows():
        if int(row["stock_end"]) < 0:
            return int(i)
    return None
