from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path("data/raw")

EQUIPMENT = [
    "wafer inspection system",
    "reticle inspection platform",
    "defect review station",
    "overlay metrology tool",
    "e-beam inspection module",
]

TOPICS = [
    "preventive maintenance",
    "fault isolation",
    "recipe tuning",
    "uptime optimization",
    "alarm recovery",
    "calibration",
    "spare parts planning",
    "process drift control",
    "yield excursion response",
    "operator handoff",
]


def build_record(i: int) -> dict:
    eq = EQUIPMENT[i % len(EQUIPMENT)]
    topic = TOPICS[i % len(TOPICS)]
    severity = ["low", "medium", "high"][i % 3]

    instruction = f"Provide a KLA field-ready response for {eq} covering {topic}."
    context = (
        f"Site=Fab-{(i % 7) + 1}, Shift={(i % 3) + 1}, "
        f"AlarmCode=KLA-{1000 + i}, Severity={severity}."
    )
    response = (
        f"For {eq}, start with safety lockout and verify alarm KLA-{1000 + i}. "
        f"Then execute the standard {topic} checklist, capture before/after metrics, "
        f"and escalate to tier-2 if recovery exceeds 30 minutes."
    )

    return {
        "id": f"kla_{i:03d}",
        "equipment": eq,
        "topic": topic,
        "severity": severity,
        "instruction": instruction,
        "context": context,
        "response": response,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, 101):
        record = build_record(i)
        path = RAW_DIR / f"kla_case_{i:03d}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
