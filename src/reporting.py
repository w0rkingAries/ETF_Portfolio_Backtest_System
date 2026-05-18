from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd

WIDTH = 150

def format_obj(obj, digits: int = 3, per_row: int | None = None) -> str:
    if obj is None:
        return ""

    if isinstance(obj, pd.Series):
        df = obj.to_frame().T
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise TypeError("物件必須是 pandas Series 或 DataFrame")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(digits)

    if per_row:
        lines = []

        cols = list(df.columns)

        for i in range(0, len(cols), per_row):
            chunk_cols = cols[i:i + per_row]
            chunk = df[chunk_cols]

            lines.append(chunk.to_string(index=False))
            lines.append("")

        return "\n".join(lines).rstrip()

    return df.to_string(index=False)

def build_horizontal_metrics(
    s: list,
    title: str = "Metrics",
    mode: str = "print",
    digits: int = 3,
    per_row: int | None = None,
) -> str:

    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print' or 'save'")

    lines = []
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    for obj in s:
        lines.append("-" * WIDTH)
        lines.append(
            format_obj(obj, digits, per_row)
        )
        lines.append("-" * WIDTH)
        lines.append("")

    
    # lines.append("=" * WIDTH)
    # lines.append(df.to_string(index=False))
    lines.append("=" * WIDTH)

    output = "\n".join(lines)

    if mode == "print" or mode == "p":
        print(output, end="\n\n")

    return output


def build_report_block(
    object: dict[str, pd.Series | pd.DataFrame],
    title: str = "Report",
    mode: str = "return",
    digits: int = 3,
) -> str:

    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print' or 'save'")

    lines = []
    
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    for name, obj in object.items():
        lines.append(f"{name}")
        lines.append("-" * WIDTH)
        lines.append(format_obj(obj, digits))
        lines.append("-" * WIDTH)
        lines.append("")

    lines.append("=" * WIDTH)
    block_text = "\n".join(lines)

    if mode == "print" or mode == "p":
        print(block_text, end="\n\n")

    return block_text


def build_regime_report(
    object: pd.DataFrame,
    title: str = "Regime Report",
    mode: str = "print",
    digits: int = 3,
) -> str:
    
    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print' or 'save'")

    lines = []
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    for strategy, sub_object in object.groupby(level=0):
        lines.append(f"{strategy}")
        lines.append("-" * WIDTH)

        display_object = sub_object.droplevel(0).copy()

        numeric_cols = display_object.select_dtypes(include="number").columns
        display_object[numeric_cols] = display_object[numeric_cols].round(digits)

        lines.append(display_object.to_string())
        lines.append("-" * WIDTH)
        lines.append("")

    lines.append("=" * WIDTH)
    report_text = "\n".join(lines)

    if mode == "print" or mode == "p":
        print(report_text, end="\n\n")

    return report_text


def build_overlay_regime_report(
    object: pd.DataFrame,
    title: str = "Overlay Regime Report",
    mode: str = "print",
    digits: int = 3,
) -> str:
    
    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print' or 'save'")

    if "overlay_col" not in object.index.names:
        return build_regime_report(
            object=object,
            title=title,
            mode=mode,
            digits=digits,
        )

    lines = []
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    overlay_cols = object.index.get_level_values("overlay_col").unique()

    for overlay_col in overlay_cols:
        lines.append(f"Overlay: {overlay_col}")
        lines.append("=" * WIDTH)
        lines.append("")

        overlay_object = object.xs(overlay_col, level="overlay_col", drop_level=True)

        for strategy, sub_object in overlay_object.groupby(level=0):
            lines.append(f"{strategy}")
            lines.append("-" * WIDTH)

            display_object = sub_object.droplevel(0).copy()

            numeric_cols = display_object.select_dtypes(include="number").columns
            display_object[numeric_cols] = display_object[numeric_cols].round(digits)

            lines.append(display_object.to_string())
            lines.append("-" * WIDTH)
            lines.append("")

    lines.append("=" * WIDTH)
    report_text = "\n".join(lines)

    if mode == "print" or mode == "p":
        print(report_text, end="\n\n")

    return report_text


def build_risk_cost_report(
    summary: pd.DataFrame,
    title: str = "Risk / Cost / Execution Summary",
    mode: str = "print",
    digits: int = 3,
) -> str:
    
    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print', 'save', 'p', or 's'")

    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")

    metric_groups = {
        "gross performance": [
            "gross_annual_return",
            "gross_volatility",
            "gross_sharpe",
            "gross_mdd",
            "gross_var_95",
        ],
        "net performance": [
            "net_annual_return",
            "net_volatility",
            "net_sharpe",
            "net_mdd",
            "net_var_95",
        ],
        "cost / execution": [
            "avg_daily_turnover",
            "annualized_turnover",
            "total_transaction_cost",
            "avg_daily_cost",
            "return_drag",
            "sharpe_drag",
        ],
    }

    lines = []
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    for strategy_name, row in summary.iterrows():
        lines.append(str(strategy_name))

        for group_name, metrics in metric_groups.items():
            available_metrics = [m for m in metrics if m in summary.columns]

            if not available_metrics:
                continue

            group_df = row[available_metrics].to_frame().T
            group_df = group_df.round(digits)

            lines.append("-" * WIDTH)
            lines.append(f"{group_name} :")
            lines.append(group_df.to_string(index=False))
            lines.append("-" * WIDTH)
            lines.append("")

    lines.append("=" * WIDTH)

    report_text = "\n".join(lines)

    if mode in {"print", "p"}:
        print(report_text, end="\n\n")

    return report_text


def export_report(
    blocks: Iterable[str],
    save_path: str | Path,
    filename: str,
    title: str = "Full Report",
    mode: str = "save",
) -> str:

    if mode not in {"save", "s"}:
        raise ValueError("mode must be 'save'")

    block_list = [b for b in blocks if b and str(b).strip()]

    lines = []
    lines.append(title)
    lines.append("#" * WIDTH)
    lines.append("")

    for i, block in enumerate(block_list, start=1):
        lines.append(f"<< Section {i} >>")
        lines.append(block)
        lines.append("")
        lines.append("#" * WIDTH)
        lines.append("")

    full_report = "\n".join(lines).rstrip()

    output_path = (Path(save_path) / filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_report, encoding="utf-8")
    print(f"Full report saved to: {output_path}")
