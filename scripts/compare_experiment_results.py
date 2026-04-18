from __future__ import annotations

import json
from pathlib import Path

THRESHOLD = 0.05


def main() -> None:
    summary_path = Path("artifacts/experiments/summary.json")
    if not summary_path.exists():
        raise FileNotFoundError("Run scripts/run_finetune_experiments.py first.")

    experiments = json.loads(summary_path.read_text(encoding="utf-8"))
    ranked = sorted(experiments, key=lambda x: x.get("delta", 0), reverse=True)
    winner = ranked[0] if ranked else None

    decision = {
        "threshold": THRESHOLD,
        "selected_experiment": winner.get("experiment") if winner else None,
        "selected_model": (winner.get("fine_tune_job") or {}).get("fine_tuned_model") if winner else None,
        "publish": bool(winner and winner.get("delta", 0) >= THRESHOLD),
        "reason": (
            f"Best delta={winner.get('delta', 0)} >= threshold={THRESHOLD}."
            if winner and winner.get("delta", 0) >= THRESHOLD
            else "No experiment met quality threshold."
        ),
        "ranking": [
            {
                "experiment": e.get("experiment"),
                "dataset_version": e.get("dataset_version"),
                "delta": e.get("delta"),
                "tuned_score": (e.get("tuned_eval") or {}).get("avg_overlap_score"),
                "baseline_score": (e.get("baseline_eval") or {}).get("avg_overlap_score"),
            }
            for e in ranked
        ],
    }

    Path("artifacts/model_registry/decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
