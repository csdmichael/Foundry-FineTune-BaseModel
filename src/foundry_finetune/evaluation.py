from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import AzureOpenAI


@dataclass
class EvalSample:
    prompt: str
    expected: str


def load_eval_samples(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"]
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
            samples.append(EvalSample(prompt=user_msg, expected=assistant_msg))
    return samples


def _tokenize(text: str) -> set[str]:
    return {t.strip(".,:;()[]{}!?").lower() for t in text.split() if t.strip()}


def overlap_score(prediction: str, expected: str) -> float:
    p = _tokenize(prediction)
    e = _tokenize(expected)
    if not e:
        return 0.0
    return len(p & e) / len(e)


def query_model(client: AzureOpenAI, model_name: str, prompt: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert KLA semiconductor equipment assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content or ""


def evaluate_model(client: AzureOpenAI, model_name: str, samples: Iterable[EvalSample]) -> dict:
    results = []
    for sample in samples:
        prediction = query_model(client, model_name, sample.prompt)
        score = overlap_score(prediction, sample.expected)
        results.append(
            {
                "prompt": sample.prompt,
                "expected": sample.expected,
                "prediction": prediction,
                "overlap_score": round(score, 4),
            }
        )

    avg = sum(r["overlap_score"] for r in results) / max(len(results), 1)
    return {"avg_overlap_score": round(avg, 4), "samples": results}
