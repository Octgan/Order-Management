"""Square CSV（ルートB）の読み込み・日付結合。"""

from square_csv.parser import (
    DayImport,
    DaySummary,
    build_square_product_label,
    is_meaningful_day,
    list_square_labels_from_matrix,
    load_uploaded_dataframe,
    match_product_name,
    parse_dual_csv_upload,
    read_uploaded_csv,
    set_custom_product_mappings,
    summarize_square_row_mapping,
)

__all__ = [
    "DayImport",
    "DaySummary",
    "build_square_product_label",
    "is_meaningful_day",
    "list_square_labels_from_matrix",
    "load_uploaded_dataframe",
    "match_product_name",
    "parse_dual_csv_upload",
    "read_uploaded_csv",
    "set_custom_product_mappings",
    "summarize_square_row_mapping",
]
