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
    ".sync_manifest.json",
]

CALENDAR_SYNC_FILES = [
    "inventory_consumption_plan.csv",
    "inventory_delivery_plan.csv",
    "inventory_actual_sales.csv",
]

SyncAction = Literal["downloaded", "uploaded", "skipped", "missing"]

_client: Any = None
_bucket: str | None = None
_last_sync_errors: list[str] = []
_last_upload_errors: list[str] = []
_last_sync_at: float = 0.0
_MTIME_TOLERANCE_SECONDS = 2.0


def is_local_data_empty() -> bool:
    """ローカルに実データが1件もない（初回起動など）。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for rel in SYNCED_FILES:
        if rel == ".sync_manifest.json":
            continue
        path = DATA_DIR / rel
        if path.exists() and path.stat().st_size > 0:
            return False
    return True


def _record_synced_state(rel_path: str) -> None:
    """クラウドとローカルが一致した状態を記録。"""
    manifest = _read_sync_manifest()
    now = time.time()
    manifest[rel_path] = now
    manifest[f"{rel_path}::cloud"] = now
    _write_sync_manifest(manifest)


def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(DATA_DIR).as_posix()
    except ValueError:
        return path.name


_last_config_error: str = ""


def _read_secret_value(section: Any, *keys: str) -> str | None:
    """Streamlit Secrets（AttrDict）から値を読む。"""
    if section is None:
        return None
    for key in keys:
        raw: Any = None
        try:
            if hasattr(section, "get"):
                raw = section.get(key)
            if raw in (None, "") and hasattr(section, key):
                raw = getattr(section, key)
            if raw in (None, "") and hasattr(section, "__getitem__"):
                raw = section[key]
        except Exception:
            raw = None
        text = str(raw or "").strip()
        if text:
            return text
    return None


def _load_supabase_section() -> Any | None:
    global _last_config_error
    try:
        import streamlit as st

        secrets = st.secrets
        if "supabase" in secrets:
            return secrets["supabase"]
        _last_config_error = "st.secrets に [supabase] セクションがありません"
    except Exception as exc:
        _last_config_error = f"Secrets 読み込みエラー: {exc}"
    return None


def _get_config() -> dict[str, str] | None:
    url = key = bucket = None
    section = _load_supabase_section()
    if section is not None:
        url = _read_secret_value(section, "url", "SUPABASE_URL")
        key = _read_secret_value(
            section,
            "key",
            "service_role_key",
            "service_role",
            "SUPABASE_KEY",
        )
        bucket = _read_secret_value(section, "bucket", "SUPABASE_BUCKET")

    if not url or not key:
        try:
            import streamlit as st

            root = st.secrets
            url = url or _read_secret_value(root, "SUPABASE_URL", "supabase_url")
            key = key or _read_secret_value(
                root,
                "SUPABASE_KEY",
                "supabase_key",
                "SUPABASE_SERVICE_ROLE_KEY",
            )
            bucket = bucket or _read_secret_value(root, "SUPABASE_BUCKET", "supabase_bucket")
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
    return _get_config() is not None


def get_cloud_setup_status() -> dict[str, Any]:
    """Secrets の設定状況（値そのものは返さない）。"""
    has_section = has_url = has_key = False
    key_length = 0
    section = _load_supabase_section()
    if section is not None:
        has_section = True
        url_val = _read_secret_value(section, "url", "SUPABASE_URL")
        key_val = _read_secret_value(
            section,
            "key",
            "service_role_key",
            "service_role",
            "SUPABASE_KEY",
        )
        has_url = bool(url_val)
        has_key = bool(key_val)
        if key_val:
            key_length = len(key_val)
    if not has_url:
        has_url = bool(os.environ.get("SUPABASE_URL", "").strip())
    if not has_key:
        env_key = os.environ.get("SUPABASE_KEY", "").strip()
        has_key = bool(env_key)
        if env_key:
            key_length = len(env_key)
    return {
        "has_section": has_section,
        "has_url": has_url,
        "has_key": has_key,
        "key_length": key_length,
        "configured": has_url and has_key,
        "error_hint": _last_config_error,
    }


def is_ephemeral_host() -> bool:
    """Streamlit Cloud など、ローカルディスクが再起動で消える環境。"""
    if os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    return bool(os.environ.get("STREAMLIT_SHARING_MODE"))


def _storage_headers(api_key: str) -> dict[str, str]:
    """Storage API 用ヘッダー。sb_secret 等の新形式キーは apiKey のみ送る。"""
    headers = {"apiKey": api_key}
    if not api_key.startswith("sb_"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_client() -> tuple[Any, str]:
    global _client, _bucket
    cfg = _get_config()
    if cfg is None:
        raise RuntimeError("Supabase is not configured")
    if _client is None:
        from storage3 import SyncStorageClient

        storage_url = f"{cfg['url'].rstrip('/')}/storage/v1"
        _client = SyncStorageClient(storage_url, _storage_headers(cfg["key"]))
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
        items = client.from_(bucket).list(prefix)
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


def _record_cloud_upload(rel_path: str) -> None:
    """クラウドへ送信済みの時刻を記録（未送信のローカル保存を古いクラウドで上書きしない）。"""
    manifest = _read_sync_manifest()
    manifest[f"{rel_path}::cloud"] = time.time()
    _write_sync_manifest(manifest)


def _has_pending_cloud_upload(rel_path: str) -> bool:
    manifest = _read_sync_manifest()
    local_ts = float(manifest.get(rel_path, 0.0))
    cloud_ts = float(manifest.get(f"{rel_path}::cloud", 0.0))
    return local_ts > cloud_ts + 0.001


def mark_calendar_data_dirty() -> None:
    """カレンダー入力 CSV をこのセッション中はクラウドで上書きしない。"""
    for rel in CALENDAR_SYNC_FILES:
        mark_session_dirty(rel)


def _record_local_write(rel_path: str) -> None:
    """ローカル保存時刻を記録（OneDrive 等で mtime がずれても新しい方を優先）。"""
    manifest = _read_sync_manifest()
    manifest[rel_path] = time.time()
    _write_sync_manifest(manifest)
    if rel_path in SYNCED_FILES:
        mark_session_dirty(rel_path)
    try:
        import streamlit as st

        recent = st.session_state.setdefault("_recent_local_writes", {})
        recent[rel_path] = manifest[rel_path]
    except Exception:
        pass


def _effective_local_mtime(rel_path: str) -> float:
    return max(_local_file_mtime(rel_path), _read_sync_manifest().get(rel_path, 0.0))


def mark_session_dirty(rel_path: str) -> None:
    """このセッションで編集したファイルをクラウドの古いコピーで上書きしない。"""
    try:
        import streamlit as st

        dirty = st.session_state.setdefault("_session_dirty_files", set())
        if not isinstance(dirty, set):
            dirty = set(dirty)
            st.session_state["_session_dirty_files"] = dirty
        dirty.add(rel_path)
    except Exception:
        pass


def clear_session_dirty(rel_path: str | None = None) -> None:
    """明示的なクラウド取得前に呼び出す。"""
    try:
        import streamlit as st

        if rel_path is None:
            st.session_state.pop("_session_dirty_files", None)
            return
        dirty = st.session_state.get("_session_dirty_files")
        if isinstance(dirty, set):
            dirty.discard(rel_path)
    except Exception:
        pass


def _is_session_dirty(rel_path: str) -> bool:
    try:
        import streamlit as st

        dirty = st.session_state.get("_session_dirty_files", set())
        return rel_path in dirty if isinstance(dirty, set) else False
    except Exception:
        return False


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
        data = client.from_(bucket).download(rel_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        _flush_path(local_path)
        _record_synced_state(rel_path)
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
        client.from_(bucket).upload(
            rel_path,
            body,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        _record_cloud_upload(rel_path)
        return True
    except Exception as exc:
        _last_upload_errors.append(f"{rel_path}: {exc}")
        return False


def sync_file_bidirectional(rel_path: str, *, force_pull: bool = False) -> SyncAction:
    """クラウド同期。通常時はローカルを消さずクラウドへ送るだけ（ローカル優先）。"""
    if not is_cloud_enabled():
        return "skipped"

    local_path = DATA_DIR / rel_path

    if force_pull:
        if download_file(rel_path, force=True):
            return "downloaded"
        if local_path.exists():
            return "uploaded" if upload_file(rel_path) else "skipped"
        return "missing"

    if local_path.exists() and local_path.stat().st_size > 0:
        return "uploaded" if upload_file(rel_path) else "skipped"

    return "downloaded" if download_file(rel_path, force=True) else "missing"


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


def _format_cloud_error(exc: Exception) -> str:
    text = str(exc)
    if "Invalid Compact JWS" in text or "Invalid JWT" in text:
        return (
            "APIキーの形式が合っていません。"
            " `sb_secret_...` を使っている場合はアプリを最新版に更新してください。"
            " または Supabase の **Legacy service_role** キー（`eyJ...` で始まる）を Secrets の key に設定してください。"
        )
    return text[:240]


def probe_cloud_connection() -> tuple[bool, str]:
    """Supabase Storage へ接続できるか確認する（秘密情報は返さない）。"""
    if not is_cloud_enabled():
        return False, "Supabase 設定が未完了です"
    try:
        client, bucket = _get_client()
        client.from_(bucket).list("", {"limit": 1})
        return True, ""
    except Exception as exc:
        return False, _format_cloud_error(exc)


def cloud_status_label() -> str:
    if not is_cloud_enabled():
        if is_ephemeral_host():
            return "一時保存（再起動で消えます）"
        return "ローカル保存（この端末のみ）"
    pending = sum(1 for rel in SYNCED_FILES if _has_pending_cloud_upload(rel))
    if _last_sync_errors or _last_upload_errors:
        return f"クラウド同期（警告 {len(_last_sync_errors) + len(_last_upload_errors)} 件）"
    if pending:
        return f"ローカル保存済み（クラウド送信待ち {pending} 件）"
    return "ローカル優先・クラウド同期 ON"


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
