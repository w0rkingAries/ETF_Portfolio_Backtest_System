from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd

WIDTH = 105

def build_horizontal_metrics(
    s: pd.Series,
    title: str = "Metrics",
    mode: str = "print",
    digits: int = 3,
) -> str:

    if mode not in {"print", "save", "p", "s"}:
        raise ValueError("mode must be 'print' or 'save'")

    df = s.to_frame().T

    # rounding
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(digits)

    lines = []

    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append(df.to_string(index=False))
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

    def format_obj(obj) -> str:
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

        return df.to_string(index=False)

    lines = []
    
    lines.append(title)
    lines.append("=" * WIDTH)
    lines.append("")

    for name, obj in object.items():
        lines.append(f"{name}")
        lines.append("-" * WIDTH)
        lines.append(format_obj(obj))
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