"""共有ストレージ: Supabase Storage で CSV を端末間同期（未設定時はローカルのみ）。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

SYNCED_FILES = [
    "products.csv",
    "product_mappings.csv",
    "daily_sales.csv",
    "inventory.csv",
    "inventory_deliveries.csv",
    "inventory_consumption_plan.csv",
    "inventory_delivery_plan.csv",
    ".data_version",
]

_client: Any = None
_bucket: str | None = None
_enabled: bool | None = None
_last_sync_errors: list[str] = []
_last_sync_at: float = 0.0


def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(DATA_DIR).as_posix()
    except ValueError:
        return path.name


def _get_config() -> dict[str, str] | None:
    url = key = bucket = None
    try:
        import streamlit as st

        cfg = st.secrets.get("supabase", {})
        if isinstance(cfg, dict):
            url = str(cfg.get("url", "") or "").strip() or None
            key = str(cfg.get("key", "") or "").strip() or None
            bucket = str(cfg.get("bucket", "") or "").strip() or None
    except Exception:
        pass

    if not url:
        url = os.environ.get("SUPABASE_URL", "").strip() or None
    if not key:
        key = os.environ.get("SUPABASE_KEY", "").strip() or None
    if not bucket:
        bucket = os.environ.get("SUPABASE_BUCKET", "").strip() or None

    if not url or not key:
        return None
    return {
        "url": url,
        "key": key,
        "bucket": bucket or "order-app-data",
    }


def is_cloud_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = _get_config() is not None
    return _enabled


def _get_client() -> tuple[Any, str]:
    global _client, _bucket
    cfg = _get_config()
    if cfg is None:
        raise RuntimeError("Supabase is not configured")
    if _client is None:
        from supabase import create_client

        _client = create_client(cfg["url"], cfg["key"])
        _bucket = cfg["bucket"]
    assert _bucket is not None
    return _client, _bucket


def _is_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not found" in text or "404" in text or "object not found" in text


def download_file(rel_path: str) -> bool:
    """クラウドからローカルへ1ファイル取得。存在しなければ False。"""
    if not is_cloud_enabled():
        return False
    client, bucket = _get_client()
    local_path = DATA_DIR / rel_path
    try:
        data = client.storage.from_(bucket).download(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise


def upload_file(rel_path: str) -> None:
    """ローカルファイルをクラウドへアップロード（上書き）。"""
    if not is_cloud_enabled():
        return
    local_path = DATA_DIR / rel_path
    if not local_path.exists():
        return
    client, bucket = _get_client()
    body = local_path.read_bytes()
    content_type = "text/plain" if rel_path.startswith(".") else "text/csv"
    client.storage.from_(bucket).upload(
        rel_path,
        body,
        file_options={"content-type": content_type, "upsert": "true"},
    )


def sync_all_from_cloud() -> tuple[int, list[str]]:
    """クラウド上の全データファイルをローカルへ取得。"""
    global _last_sync_errors, _last_sync_at
    if not is_cloud_enabled():
        return 0, []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    errors: list[str] = []
    for rel in SYNCED_FILES:
        try:
            if download_file(rel):
                count += 1
        except Exception as exc:
            errors.append(f"{rel}: {exc}")

    _last_sync_errors = errors
    _last_sync_at = time.time()
    return count, errors


def push_all_to_cloud() -> tuple[int, list[str]]:
    """ローカルの全データファイルをクラウドへ送信（初回移行用）。"""
    if not is_cloud_enabled():
        return 0, []

    count = 0
    errors: list[str] = []
    for rel in SYNCED_FILES:
        local_path = DATA_DIR / rel
        if not local_path.exists():
            continue
        try:
            upload_file(rel)
            count += 1
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
    return count, errors


def ensure_cloud_sync(*, force: bool = False, ttl_seconds: int = 45) -> int:
    """セッション中は一定間隔でクラウドから取得（force=True で即時）。戻り値=取得したファイル数。"""
    if not is_cloud_enabled():
        return 0
    try:
        import streamlit as st

        last = float(st.session_state.get("_cloud_sync_at", 0.0))
        now = time.time()
        if force or now - last >= ttl_seconds:
            count, _errors = sync_all_from_cloud()
            st.session_state["_cloud_sync_at"] = now
            return count
    except Exception:
        if force or time.time() - _last_sync_at >= ttl_seconds:
            count, _errors = sync_all_from_cloud()
            return count
    return 0


def cloud_status_label() -> str:
    if not is_cloud_enabled():
        return "ローカル保存（この端末のみ）"
    if _last_sync_errors:
        return f"クラウド同期（警告 {len(_last_sync_errors)} 件）"
    return "クラウド同期 ON"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    defaults: dict[str, Any] = {"encoding": "utf-8-sig"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"index": False, "encoding": "utf-8-sig"}
    options.update(kwargs)
    df.to_csv(path, **options)
    upload_file(_rel_path(path))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    upload_file(_rel_path(path))
