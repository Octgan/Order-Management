"""共有ストレージ: Supabase Storage で CSV を端末間同期（未設定時はローカルのみ）。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNC_MANIFEST_PATH = DATA_DIR / ".sync_manifest.json"

SYNCED_FILES = [
    "products.csv",
    "product_mappings.csv",
    "daily_sales.csv",
    "inventory.csv",
    "inventory_deliveries.csv",
    "inventory_consumption_plan.csv",
    "inventory_delivery_plan.csv",
    "inventory_actual_sales.csv",
    ".data_version",
]

SyncAction = Literal["downloaded", "uploaded", "skipped", "missing"]

_client: Any = None
_bucket: str | None = None
_enabled: bool | None = None
_last_sync_errors: list[str] = []
_last_upload_errors: list[str] = []
_last_sync_at: float = 0.0
_MTIME_TOLERANCE_SECONDS = 2.0
_WRITE_PROTECT_SECONDS = 180.0


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


def is_ephemeral_host() -> bool:
    """Streamlit Cloud など、ローカルディスクが再起動で消える環境。"""
    if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    return bool(os.environ.get("STREAMLIT_SHARING_MODE"))


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


def _parse_cloud_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).timestamp()
    except Exception:
        return None


def _cloud_file_mtime(rel_path: str) -> float | None:
    """Supabase Storage 上のファイル更新時刻（UNIX秒）。"""
    if not is_cloud_enabled():
        return None
    client, bucket = _get_client()
    name = Path(rel_path).name
    parent = Path(rel_path).parent.as_posix()
    prefix = "" if parent in ("", ".") else parent
    try:
        items = client.storage.from_(bucket).list(prefix)
        for item in items or []:
            if str(item.get("name")) != name:
                continue
            for key in ("updated_at", "created_at", "last_accessed_at"):
                if ts := _parse_cloud_timestamp(item.get(key)):
                    return ts
    except Exception:
        return None
    return None


def _local_file_mtime(rel_path: str) -> float:
    local_path = DATA_DIR / rel_path
    if not local_path.exists():
        return 0.0
    return local_path.stat().st_mtime


def _read_sync_manifest() -> dict[str, float]:
    if not SYNC_MANIFEST_PATH.exists():
        return {}
    try:
        raw = json.loads(SYNC_MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k): float(v) for k, v in raw.items()}
    except Exception:
        return {}


def _write_sync_manifest(manifest: dict[str, float]) -> None:
    SYNC_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


def _record_local_write(rel_path: str) -> None:
    """ローカル保存時刻を記録（OneDrive 等で mtime がずれても新しい方を優先）。"""
    manifest = _read_sync_manifest()
    manifest[rel_path] = time.time()
    _write_sync_manifest(manifest)
    try:
        import streamlit as st

        recent = st.session_state.setdefault("_recent_local_writes", {})
        recent[rel_path] = manifest[rel_path]
    except Exception:
        pass


def _effective_local_mtime(rel_path: str) -> float:
    return max(_local_file_mtime(rel_path), _read_sync_manifest().get(rel_path, 0.0))


def _is_recently_written(rel_path: str) -> bool:
    try:
        import streamlit as st

        recent_ts = float(st.session_state.get("_recent_local_writes", {}).get(rel_path, 0.0))
        if time.time() - recent_ts < _WRITE_PROTECT_SECONDS:
            return True
    except Exception:
        pass
    manifest_ts = _read_sync_manifest().get(rel_path, 0.0)
    return time.time() - manifest_ts < _WRITE_PROTECT_SECONDS


def _set_persist_notice(message: str, level: str = "info") -> None:
    try:
        import streamlit as st

        st.session_state["_persist_notice"] = {"message": message, "level": level}
    except Exception:
        pass


def set_persist_notice(message: str, level: str = "info") -> None:
    _set_persist_notice(message, level)


def pop_persist_notice() -> dict[str, str] | None:
    try:
        import streamlit as st

        notice = st.session_state.pop("_persist_notice", None)
        if isinstance(notice, dict) and notice.get("message"):
            return notice
    except Exception:
        pass
    return None


def _flush_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def download_file(rel_path: str, *, force: bool = False) -> bool:
    """クラウドからローカルへ1ファイル取得。存在しなければ False。"""
    if not is_cloud_enabled():
        return False
    client, bucket = _get_client()
    local_path = DATA_DIR / rel_path
    if not force and local_path.exists():
        cloud_mtime = _cloud_file_mtime(rel_path)
        local_mtime = _effective_local_mtime(rel_path)
        if cloud_mtime is not None and cloud_mtime <= local_mtime + _MTIME_TOLERANCE_SECONDS:
            return False
    try:
        data = client.storage.from_(bucket).download(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        _flush_path(local_path)
        return True
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise


def upload_file(rel_path: str) -> bool:
    """ローカルファイルをクラウドへアップロード（上書き）。"""
    global _last_upload_errors
    if not is_cloud_enabled():
        return True
    local_path = DATA_DIR / rel_path
    if not local_path.exists():
        return False
    client, bucket = _get_client()
    body = local_path.read_bytes()
    content_type = "text/plain" if rel_path.startswith(".") else "text/csv"
    try:
        client.storage.from_(bucket).upload(
            rel_path,
            body,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return True
    except Exception as exc:
        _last_upload_errors.append(f"{rel_path}: {exc}")
        return False


def sync_file_bidirectional(rel_path: str, *, force_pull: bool = False) -> SyncAction:
    """ローカルとクラウドを更新日時で突き合わせ、新しい方を優先して同期。"""
    if not is_cloud_enabled():
        return "skipped"

    local_path = DATA_DIR / rel_path
    local_mtime = _effective_local_mtime(rel_path)

    if force_pull:
        if _is_recently_written(rel_path) and local_path.exists():
            return "uploaded" if upload_file(rel_path) else "skipped"
        if download_file(rel_path, force=True):
            return "downloaded"
        if local_path.exists():
            return "uploaded" if upload_file(rel_path) else "skipped"
        return "missing"

    cloud_mtime = _cloud_file_mtime(rel_path)

    if _is_recently_written(rel_path) and local_path.exists():
        return "uploaded" if upload_file(rel_path) else "skipped"

    if cloud_mtime is None:
        if local_path.exists():
            return "uploaded" if upload_file(rel_path) else "skipped"
        return "missing"

    if not local_path.exists():
        return "downloaded" if download_file(rel_path, force=True) else "missing"

    if cloud_mtime > local_mtime + _MTIME_TOLERANCE_SECONDS:
        return "downloaded" if download_file(rel_path, force=True) else "skipped"
    if local_mtime > cloud_mtime + _MTIME_TOLERANCE_SECONDS:
        return "uploaded" if upload_file(rel_path) else "skipped"
    return "skipped"


def sync_all_from_cloud(*, force_pull: bool = False) -> tuple[int, list[str]]:
    """クラウドとローカルのデータファイルを同期（双方向・新しい方を優先）。"""
    global _last_sync_errors, _last_sync_at
    if not is_cloud_enabled():
        return 0, []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    errors: list[str] = []
    for rel in SYNCED_FILES:
        try:
            action = sync_file_bidirectional(rel, force_pull=force_pull)
            if action in ("downloaded", "uploaded"):
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
            if upload_file(rel):
                count += 1
            else:
                errors.append(f"{rel}: upload failed")
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
    return count, errors


def ensure_cloud_sync(
    *,
    force: bool = False,
    force_pull: bool = False,
    ttl_seconds: int = 45,
) -> int:
    """一定間隔でクラウド同期。戻り値=更新したファイル数。"""
    if not is_cloud_enabled():
        return 0
    try:
        import streamlit as st

        last = float(st.session_state.get("_cloud_sync_at", 0.0))
        now = time.time()
        if force or force_pull or now - last >= ttl_seconds:
            count, _errors = sync_all_from_cloud(force_pull=force_pull)
            st.session_state["_cloud_sync_at"] = now
            return count
    except Exception:
        if force or force_pull or time.time() - _last_sync_at >= ttl_seconds:
            count, _errors = sync_all_from_cloud(force_pull=force_pull)
            return count
    return 0


def cloud_status_label() -> str:
    if not is_cloud_enabled():
        if is_ephemeral_host():
            return "一時保存（再起動で消えます）"
        return "ローカル保存（この端末のみ）"
    if _last_sync_errors or _last_upload_errors:
        return f"クラウド同期（警告 {len(_last_sync_errors) + len(_last_upload_errors)} 件）"
    return "クラウド同期 ON"


def last_upload_errors() -> list[str]:
    return list(_last_upload_errors)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    defaults: dict[str, Any] = {"encoding": "utf-8-sig"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> bool:
    """CSV をローカルへ保存し、クラウド設定時はアップロードも行う。戻り値=クラウド送信成功。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"index": False, "encoding": "utf-8-sig"}
    options.update(kwargs)
    rel = _rel_path(path)
    tmp_path = path.with_suffix(path.suffix + ".part")
    df.to_csv(tmp_path, **options)
    _flush_path(tmp_path)
    tmp_path.replace(path)
    _flush_path(path)
    _record_local_write(rel)

    if not is_cloud_enabled():
        if is_ephemeral_host():
            _set_persist_notice(
                "データはこのセッション中のみ保持されます。"
                " 永続保存には Supabase の設定が必要です（secrets.toml を参照）。",
                "warning",
            )
        return True

    uploaded = upload_file(rel)
    if uploaded:
        return True

    _set_persist_notice(
        f"「{path.name}」はこの端末に保存しましたが、クラウドへの送信に失敗しました。"
        " 別端末では反映されず、再起動後に消える可能性があります。"
        " サイドバーの「この端末のデータをクラウドへ送信」をお試しください。",
        "warning",
    )
    return False


def write_text(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.write_text(text, encoding="utf-8")
    _flush_path(tmp_path)
    tmp_path.replace(path)
    _flush_path(path)
    _record_local_write(_rel_path(path))
    if not is_cloud_enabled():
        return True
    return upload_file(_rel_path(path))
