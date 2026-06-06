"""MTG PDF 本体（reportlab はここだけで読み込む）。"""

from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mtg_report import MtgReportContext

JP_FONT = "HeiseiKakuGo-W5"


def build_mtg_report_pdf_impl(ctx: MtgReportContext) -> bytes:
    """reportlab で PDF を生成。"""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    page_w, _page_h = A4
    margin = 18 * mm

    if JP_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(JP_FONT))

    base = getSampleStyleSheet()
    styles = {
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
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=JP_FONT,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0f3460"),
            spaceBefore=8,
            spaceAfter=4,
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

    def make_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
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
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8f9fa")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return t

    def kv_table(rows: list[tuple[str, str]]) -> Table:
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

    def add_chart(story: list[Any], img_bytes: bytes | None, caption: str) -> None:
        if not img_bytes:
            return
        try:
            story.append(Spacer(1, 4))
            img = Image(io.BytesIO(img_bytes))
            max_w = page_w - 2 * margin
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

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="MTGレポート_全フード",
    )
    story: list[Any] = []
    created = datetime.now().strftime("%Y/%m/%d %H:%M")
    charts = ctx.chart_images or {}

    story.append(Paragraph(ctx.app_title, styles["title"]))
    story.append(Paragraph("MTG レポート（全フード・売上・発注）", styles["h1"]))
    story.append(Spacer(1, 6))
    story.append(
        kv_table(
            [
                ("対象", "全フード商品"),
                ("集計期間", ctx.period_label),
                ("作成日時", created),
            ]
        )
    )
    if ctx.memo.strip():
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"メモ: {ctx.memo.strip()}", styles["body"]))

    story.append(Paragraph("1. 店舗サマリー（最新営業日）", styles["h1"]))
    story.append(
        kv_table(
            [
                ("対象日", ctx.latest_store_label),
                ("店舗純売上（税抜）", f"¥{ctx.latest_store_sales:,}"),
            ]
        )
    )

    story.append(Paragraph("2. 全フード選択率", styles["h1"]))
    story.append(
        kv_table(
            [
                ("全フード選択率", f"{ctx.all_foods_rate:.2%}"),
                ("全フード販売数合計", f"{ctx.all_foods_units:,} 個"),
                ("期間店舗客数合計", f"{ctx.period_customers:,} 人"),
                ("平均客数", f"{ctx.customer_stats.get('avg', 0):,.1f} 人/日"),
                ("期間合計客数", f"{int(ctx.customer_stats.get('total', 0)):,} 人"),
            ]
        )
    )
    add_chart(story, charts.get("selection_pie"), "全フード選択率（商品別内訳）")

    if not ctx.food_breakdown.empty:
        bd_rows = [["フード商品", "期間販売数", "選択率"]]
        for _, row in ctx.food_breakdown.iterrows():
            bd_rows.append(
                [
                    str(row["product_name"]),
                    f"{int(row['units']):,} 個",
                    f"{float(row['selection_rate']):.2%}",
                ]
            )
        story.append(Spacer(1, 6))
        story.append(make_table(bd_rows, col_widths=[50 * mm, 45 * mm, 45 * mm]))

    add_chart(story, charts.get("customer_trend"), "店舗客数の推移")

    story.append(Paragraph("3. 発注予測の共通前提", styles["h1"]))
    story.append(
        kv_table(
            [
                ("予測客数", f"{ctx.predicted_customers:,} 人"),
                ("天気・イベント補正", f"{ctx.correction_pct:+d}%（×{ctx.correction_factor:.2f}）"),
                ("冷蔵収容上限", f"{ctx.cold_storage_capacity:,} 個"),
            ]
        )
    )
    story.append(
        Paragraph(
            "各フードの推奨発注量 ＝ 予測客数 × 当該フードの選択率 × 補正係数 − 現在在庫",
            styles["small"],
        )
    )

    for idx, food in enumerate(ctx.food_sections, start=1):
        story.append(PageBreak())
        story.append(Paragraph(f"4-{idx}. {food.product_name}", styles["h1"]))
        story.append(Paragraph("売上・販売", styles["h2"]))
        story.append(
            kv_table(
                [
                    ("最新日", food.latest_label),
                    ("最新日販売数", f"{food.latest_units:,} 個"),
                    ("単品売上高（税抜）", f"¥{food.latest_single_item_sales:,}"),
                    ("前週同曜日販売数", f"{food.lw_sales:,} 個"),
                    ("過去4週同曜日平均", f"{food.avg_4w:,.1f} 個"),
                    ("フード選択率", f"{food.selection_rate:.2%}"),
                    ("期間合計販売数", f"{food.period_units:,} 個"),
                    ("平均販売個数", f"{food.period_stats.get('avg', 0):,.1f} 個/日"),
                    ("平均単品売上高", f"¥{food.single_item_stats.get('avg', 0):,.0f}"),
                ]
            )
        )

        if not food.weekday_units.empty:
            story.append(Spacer(1, 4))
            story.append(Paragraph("曜日別の平均販売個数", styles["h2"]))
            wd_rows = [["曜日", "平均", "日数", "合計"]]
            for _, row in food.weekday_units.iterrows():
                wd_rows.append(
                    [
                        str(row["曜日"]),
                        f"{float(row['avg_units']):,.1f} 個",
                        f"{int(row['days'])} 日",
                        f"{int(row['total_units']):,} 個",
                    ]
                )
            story.append(make_table(wd_rows, col_widths=[22 * mm, 38 * mm, 30 * mm, 40 * mm]))

        theory = int(ctx.predicted_customers * food.selection_rate * ctx.correction_factor)
        cap_note = ""
        if food.recommended_order > ctx.cold_storage_capacity:
            cap_note = "※ 推奨量が冷蔵収容上限を超えています。"
        elif food.recommended_order > ctx.cold_storage_capacity * 0.85:
            cap_note = "※ 収容上限の85%超。"

        story.append(Paragraph("発注予測", styles["h2"]))
        story.append(
            kv_table(
                [
                    ("現在在庫", f"{food.current_stock:,} 個"),
                    ("理論需要数", f"{theory:,} 個"),
                    ("推奨発注量", f"{food.recommended_order:,} 個"),
                ]
            )
        )
        if cap_note:
            story.append(Paragraph(cap_note, styles["body"]))

        if food.inventory_summary:
            inv = food.inventory_summary
            story.append(Paragraph("在庫見込み", styles["h2"]))
            story.append(
                kv_table(
                    [
                        ("安全在庫", f"{inv.get('safety_stock', 0):,} 個"),
                        ("収納MAX", f"{inv.get('max_stock', 0):,} 個"),
                        ("計画の平均消費", f"{inv.get('avg_use', 0):,.1f} 個/日"),
                        ("期間内の最低在庫", f"{inv.get('min_stock', 0):,} 個"),
                        ("在庫切れ予測", inv.get("stockout_label", "—")),
                        (
                            f"{inv.get('horizon', 0)}日後の予想在庫",
                            f"{inv.get('end_stock', 0):,} 個",
                        ),
                    ]
                )
            )
            add_chart(story, charts.get(f"inventory_{food.product_id}"), "消費と予想在庫")

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "※ 数値はアプリ登録データに基づく参考値です。MTG では前提条件をご確認ください。",
            styles["small"],
        )
    )

    doc.build(story)
    return buf.getvalue()
