from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path("data/raw")
SETS_DIR = Path("data/sets")

SYSTEM_PROMPT = "You are an expert KLA semiconductor equipment support assistant."


def to_chat_example(record: dict) -> dict:
    user_content = f"{record['instruction']} Context: {record['context']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["response"]},
        ]
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split(rows: list[dict], train_ratio: float = 0.8) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + max(1, int((n - train_end) / 2))
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def main() -> None:
    all_records = []
    for p in sorted(RAW_DIR.glob("kla_case_*.json")):
        all_records.append(json.loads(p.read_text(encoding="utf-8")))

    set_a_records = [r for i, r in enumerate(all_records) if i % 2 == 0]
    set_b_records = [r for i, r in enumerate(all_records) if i % 2 == 1]

    for set_name, records in {"set_a": set_a_records, "set_b": set_b_records}.items():
        rows = [to_chat_example(r) for r in records]
        train_rows, val_rows, eval_rows = split(rows)
        write_jsonl(SETS_DIR / set_name / "train.jsonl", train_rows)
        write_jsonl(SETS_DIR / set_name / "validation.jsonl", val_rows)
        write_jsonl(SETS_DIR / set_name / "eval.jsonl", eval_rows)

    manifest = {
        "datasets": [
            {"name": "set_a", "dataset_version": "kla-v1", "records": len(set_a_records)},
            {"name": "set_b", "dataset_version": "kla-v2", "records": len(set_b_records)},
        ]
    }
    Path("artifacts/datasets/manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
