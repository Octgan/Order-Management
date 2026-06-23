"""在庫管理（Streamlit Cloud 向けパッケージ）。"""

from __future__ import annotations

__all__ = [
    "CALENDAR_PAST_DAYS",
    "CALENDAR_TOTAL_DAYS",
    "DEFAULT_MAX_STOCK",
    "add_delivery",
    "apply_calendar_inputs",
    "apply_planned_consumption",
    "build_inventory_calendar_projection",
    "build_inventory_projection",
    "calendar_date_range",
    "clear_actual_sales",
    "clear_consumption_plan",
    "clear_delivery_plan",
    "days_until_stockout",
    "delete_delivery",
    "get_delivery_weekdays",
    "init_actual_sales_csv",
    "init_consumption_plan_csv",
    "init_deliveries_csv",
    "init_delivery_plan_csv",
    "init_inventory_csv",
    "load_deliveries_df",
    "load_inventory_df",
    "load_manual_actual_sales",
    "load_manual_planned_use",
    "replace_actual_sales_for_period",
    "replace_delivery_plan_for_period",
    "save_consumption_plan",
    "save_delivery_plan",
    "save_delivery_weekdays",
    "save_product_inventory",
    "sync_inventory_products",
    "weekday_labels_text",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from food_stock import logic

    return getattr(logic, name)
