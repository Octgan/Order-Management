"""MTG用 PDF レポート生成（売上・発注・在庫）。"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

JP_FONT = "HeiseiKakuGo-W5"
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


@dataclass
class MtgReportContext:
    """PDF に載せる集計データ。"""

    app_title: str
    product_name: str
    period_label: str
    period_start: date
    period_end: date
    latest_label: str
    latest_units: int
    latest_store_sales: int
    latest_single_item_sales: int
    lw_sales: int
    avg_4w: float
    selection_rate: float
    period_units: int
    period_customers: int
    period_stats: dict[str, Any]
    customer_stats: dict[str, Any]
    single_item_stats: dict[str, Any]
    weekday_units: pd.DataFrame
    predicted_customers: int
    correction_pct: int
    current_stock: int
    recommended_order: int
    correction_factor: float
    cold_storage_capacity: int
    inventory_summary: dict[str, Any] | None = None
    chart_images: dict[str, bytes] | None = None
    memo: str = ""


def _register_japanese_font() -> None:
    if JP_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(JP_FONT))


def _styles() -> dict[str, ParagraphStyle]:
    _register_japanese_font()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=JP_FONT,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=JP_FONT,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1a1a2e"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName=JP_FONT,
            fontSize=10,
            leading=14,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName=JP_FONT,
            fontSize=8,
            leading=11,
            textColor=colors.grey,
        ),
    }


def _table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), JP_FONT, 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    data = [[k, v] for k, v in rows]
    t = Table(data, colWidths=[55 * mm, 105 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), JP_FONT, 9),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f2f5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _add_chart(story: list[Any], styles: dict[str, ParagraphStyle], img_bytes: bytes | None, caption: str) -> None:
    if not img_bytes:
        return
    try:
        story.append(Spacer(1, 4))
        img = Image(io.BytesIO(img_bytes))
        max_w = PAGE_W - 2 * MARGIN
        ratio = img.imageHeight / img.imageWidth if img.imageWidth else 1
        img.drawWidth = max_w
        img.drawHeight = max_w * ratio
        if img.drawHeight > 95 * mm:
            img.drawHeight = 95 * mm
            img.drawWidth = img.drawHeight / ratio
        story.append(img)
        story.append(Paragraph(caption, styles["small"]))
    except Exception:
        pass


def plotly_fig_to_png(fig: Any, width: int = 900, height: int = 420) -> bytes | None:
    """Plotly 図を PNG に（kaleido が無い環境では None）。"""
    try:
        return fig.to_image(format="png", width=width, height=height, engine="kaleido")
    except Exception:
        try:
            return fig.to_image(format="png", width=width, height=height)
        except Exception:
            return None


def build_mtg_report_pdf(ctx: MtgReportContext) -> bytes:
    """MTG 用 PDF のバイナリを生成。"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"MTGレポート_{ctx.product_name}",
    )
    styles = _styles()
    story: list[Any] = []
    created = datetime.now().strftime("%Y/%m/%d %H:%M")

    story.append(Paragraph(ctx.app_title, styles["title"]))
    story.append(Paragraph("MTG レポート（売上・発注）", styles["h1"]))
    story.append(Spacer(1, 6))
    story.append(
        _kv_table(
            [
                ("対象商品", ctx.product_name),
                ("集計期間", ctx.period_label),
                ("作成日時", created),
            ]
        )
    )
    if ctx.memo.strip():
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"メモ: {ctx.memo.strip()}", styles["body"]))

    # --- 売上サマリー（最新日） ---
    story.append(Paragraph("1. 売上サマリー（最新営業日）", styles["h1"]))
    story.append(
        _kv_table(
            [
                ("対象日", ctx.latest_label),
                ("店舗純売上（税抜）", f"¥{ctx.latest_store_sales:,}"),
                ("販売数", f"{ctx.latest_units:,} 個"),
                ("単品売上高（税抜）", f"¥{ctx.latest_single_item_sales:,}"),
                ("前週同曜日販売数", f"{ctx.lw_sales:,} 個"),
                ("過去4週同曜日平均", f"{ctx.avg_4w:,.1f} 個"),
            ]
        )
    )

    # --- フード選択率・期間分析 ---
    story.append(Paragraph("2. フード選択率・期間分析", styles["h1"]))
    story.append(
        _kv_table(
            [
                ("フード選択率", f"{ctx.selection_rate:.2%}"),
                ("期間販売数合計", f"{ctx.period_units:,} 個"),
                ("期間店舗客数合計", f"{ctx.period_customers:,} 人"),
                ("平均販売個数", f"{ctx.period_stats.get('avg', 0):,.1f} 個/日"),
                ("期間合計販売数", f"{int(ctx.period_stats.get('total', 0)):,} 個"),
                ("平均客数", f"{ctx.customer_stats.get('avg', 0):,.1f} 人/日"),
                ("期間合計客数", f"{int(ctx.customer_stats.get('total', 0)):,} 人"),
                (
                    "平均単品売上高",
                    f"¥{ctx.single_item_stats.get('avg', 0):,.0f}",
                ),
            ]
        )
    )
    charts = ctx.chart_images or {}
    _add_chart(story, styles, charts.get("selection_pie"), "フード選択率（丸グラフ）")
    _add_chart(story, styles, charts.get("weekday_bar"), "曜日別の平均販売個数")
    _add_chart(story, styles, charts.get("sales_trend"), "販売数の推移")
    _add_chart(story, styles, charts.get("customer_trend"), "店舗客数の推移")

    if not ctx.weekday_units.empty:
        story.append(Spacer(1, 6))
        story.append(Paragraph("曜日別の平均販売個数（表）", styles["body"]))
        wd_rows = [["曜日", "平均販売数", "集計日数", "合計"]]
        for _, row in ctx.weekday_units.iterrows():
            wd_rows.append(
                [
                    str(row["曜日"]),
                    f"{float(row['avg_units']):,.1f} 個",
                    f"{int(row['days'])} 日",
                    f"{int(row['total_units']):,} 個",
                ]
            )
        story.append(_table(wd_rows, col_widths=[25 * mm, 40 * mm, 35 * mm, 40 * mm]))

    # --- 発注予測 ---
    story.append(Paragraph("3. 発注予測", styles["h1"]))
    theory = int(ctx.predicted_customers * ctx.selection_rate * ctx.correction_factor)
    cap_note = ""
    if ctx.recommended_order > ctx.cold_storage_capacity:
        cap_note = f"※ 推奨量が冷蔵収容上限（{ctx.cold_storage_capacity:,} 個）を超えています。"
    elif ctx.recommended_order > ctx.cold_storage_capacity * 0.85:
        cap_note = f"※ 収容上限の85%超（上限 {ctx.cold_storage_capacity:,} 個）"

    story.append(
        _kv_table(
            [
                ("予測客数", f"{ctx.predicted_customers:,} 人"),
                ("フード選択率", f"{ctx.selection_rate:.4f}"),
                ("天気・イベント補正", f"{ctx.correction_pct:+d}%（×{ctx.correction_factor:.2f}）"),
                ("現在在庫", f"{ctx.current_stock:,} 個"),
                ("理論需要数", f"{theory:,} 個"),
                ("推奨発注量", f"{ctx.recommended_order:,} 個"),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "推奨発注量 ＝ 予測客数 × フード選択率 × 補正係数 − 現在在庫",
            styles["small"],
        )
    )
    if cap_note:
        story.append(Paragraph(cap_note, styles["body"]))

    # --- 在庫見込み ---
    if ctx.inventory_summary:
        inv = ctx.inventory_summary
        story.append(Paragraph("4. 在庫見込み", styles["h1"]))
        story.append(
            _kv_table(
                [
                    ("現在の在庫", f"{inv.get('current_stock', 0):,} 個"),
                    ("安全在庫", f"{inv.get('safety_stock', 0):,} 個"),
                    ("表示日数", f"{inv.get('horizon', 0)} 日"),
                    ("計画の平均消費", f"{inv.get('avg_use', 0):,.1f} 個/日"),
                    ("期間内の最低在庫", f"{inv.get('min_stock', 0):,} 個"),
                    (
                        "在庫切れ予測",
                        inv.get("stockout_label", "—"),
                    ),
                    (
                        f"{inv.get('horizon', 0)}日後の予想在庫",
                        f"{inv.get('end_stock', 0):,} 個",
                    ),
                ]
            )
        )
        _add_chart(story, styles, charts.get("inventory"), "消費と予想在庫（カレンダー）")

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "※ 数値はアプリ登録データに基づく参考値です。MTG では前提条件をご確認ください。",
            styles["small"],
        )
    )

    doc.build(story)
    return buf.getvalue()
