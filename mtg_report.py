"""MTG用 PDF レポート（データ構造・PNG変換。reportlab は PDF 生成時のみ）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class FoodReportSection:
    """PDF 内のフード1件分。"""

    product_name: str
    product_id: str
    selection_rate: float
    period_units: int
    latest_label: str
    latest_units: int
    latest_single_item_sales: int
    lw_sales: int
    avg_4w: float
    period_stats: dict[str, Any]
    single_item_stats: dict[str, Any]
    weekday_units: pd.DataFrame
    recommended_order: int
    current_stock: int
    inventory_summary: dict[str, Any] | None = None


@dataclass
class MtgReportContext:
    """PDF に載せる集計データ（全フード）。"""

    app_title: str
    period_label: str
    period_start: date
    period_end: date
    latest_store_label: str
    latest_store_sales: int
    all_foods_units: int
    all_foods_rate: float
    period_customers: int
    food_breakdown: pd.DataFrame
    customer_stats: dict[str, Any]
    predicted_customers: int
    correction_pct: int
    correction_factor: float
    cold_storage_capacity: int
    food_sections: list[FoodReportSection] = field(default_factory=list)
    chart_images: dict[str, bytes] | None = None
    memo: str = ""


def plotly_fig_to_png(fig: Any, width: int = 900, height: int = 420) -> bytes | None:
    """Plotly 図を PNG に（エンジン未導入時は None）。"""
    try:
        return fig.to_image(format="png", width=width, height=height)
    except Exception:
        return None


def build_mtg_report_pdf(ctx: MtgReportContext) -> bytes:
    """MTG 用 PDF のバイナリを生成（全フード）。"""
    try:
        from mtg_report_pdf import build_mtg_report_pdf_impl

        return build_mtg_report_pdf_impl(ctx)
    except ImportError as exc:
        raise ImportError(
            "PDF生成に reportlab が必要です。requirements.txt を確認してください。"
        ) from exc
