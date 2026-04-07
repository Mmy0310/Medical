#!/usr/bin/env python3
"""Discover and classify multidisease output folders without hardcoding folder count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ROLE_RULES = [
    ("compare", "对照/背景信号通道"),
    ("hard_exclude", "机制解释通道"),
    ("multiseed", "稳定性复现"),
    ("recover_probe", "恢复性调参"),
    ("clinical_v2", "预测主通道"),
]


def classify_output_dir(name: str) -> str:
    for token, role in ROLE_RULES:
        if token in name:
            return role
    return "其他实验目录"


def discover(base_dir: Path, prefix: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(base_dir.glob(f"{prefix}*")):
        if not path.is_dir():
            continue
        rows.append(
            {
                "name": path.name,
                "path": str(path.as_posix()),
                "role": classify_output_dir(path.name),
            }
        )
    return rows


def to_markdown(rows: List[Dict[str, str]]) -> str:
    lines = [
        "# 动态输出目录索引",
        "",
        "| 目录名 | 角色 | 路径 |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['role']} | {row['path']} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="examples/tabular",
        help="Base directory that contains outputs folders.",
    )
    parser.add_argument(
        "--prefix",
        default="outputs_project_multidisease",
        help="Folder name prefix used for discovery.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "md"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--write",
        default="",
        help="Optional file path to write results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)
    rows = discover(base_dir=base_dir, prefix=args.prefix)

    if args.format == "md":
        payload = to_markdown(rows)
    else:
        payload = json.dumps(rows, ensure_ascii=False, indent=2)

    if args.write:
        out_path = Path(args.write)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
