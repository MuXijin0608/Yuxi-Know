from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RELATION_MAP = {
    "consist_of": "包含",
    "part_of": "属于",
    "working_mode": "工作方式",
    "failure_alert": "故障预警",
    "related_to": "相关",
    "lead_to": "导致",
}


def extract_relations(input_dir: Path, output_file: Path) -> int:
    input_dir = input_dir.resolve()
    output_file = output_file.resolve()

    relations: list[dict[str, str]] = []

    for xlsx_path in sorted(input_dir.glob("*.xlsx")):
        df = pd.read_excel(xlsx_path, sheet_name=1)
        for _, row in df.iterrows():
            head = row.get("startName")
            tail = row.get("endName")
            rel = row.get("type")
            if pd.isna(head) or pd.isna(tail) or pd.isna(rel):
                continue
            head_text = str(head).strip()
            tail_text = str(tail).strip()
            rel_text = str(rel).strip()
            if not head_text or not tail_text or not rel_text:
                continue
            rel_text = RELATION_MAP.get(rel_text, rel_text)
            relations.append({"h": head_text, "t": tail_text, "r": rel_text})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for item in relations:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(relations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract relation sheet (2nd sheet) from XLSX files into JSONL.")
    parser.add_argument(
        "--input-dir",
        default="test/data/mydata/doc",
        help="Directory containing XLSX files.",
    )
    parser.add_argument(
        "--output",
        default="test/data/mydata/relations.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    total = extract_relations(Path(args.input_dir), Path(args.output))
    print(f"Wrote {total} relations to {args.output}")


if __name__ == "__main__":
    main()
