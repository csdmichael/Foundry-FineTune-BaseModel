from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

from src.foundry_finetune.evaluation import evaluate_model, load_eval_samples


@dataclass
class ExperimentConfig:
    name: str
    dataset_version: str
    description: str
    train_blob: str
    validation_blob: str
    evaluation_blob: str


def load_config(path: Path) -> ExperimentConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(**cfg)


def blob_download(blob_service: BlobServiceClient, container: str, blob_name: str, target: Path) -> Path:
    client = blob_service.get_blob_client(container=container, blob=blob_name)
    target.write_bytes(client.download_blob().readall())
    return target


def wait_for_job(client: AzureOpenAI, job_id: str, interval: int = 30) -> dict:
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = getattr(job, "status", "unknown")
        if status in {"succeeded", "failed", "cancelled"}:
            return job.model_dump() if hasattr(job, "model_dump") else dict(job)
        print(f"Job {job_id} status: {status}")
        time.sleep(interval)


def run_experiment(config_path: Path) -> dict:
    cfg = load_config(config_path)
    container = os.environ.get("AZURE_STORAGE_CONTAINER", "kla-finetune")
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        credential=credential,
    )

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        train_file = blob_download(blob_service, container, cfg.train_blob, tmp / "train.jsonl")
        val_file = blob_download(blob_service, container, cfg.validation_blob, tmp / "validation.jsonl")
        eval_file = blob_download(blob_service, container, cfg.evaluation_blob, tmp / "eval.jsonl")

        train_uploaded = client.files.create(file=train_file.open("rb"), purpose="fine-tune")
        val_uploaded = client.files.create(file=val_file.open("rb"), purpose="fine-tune")

        base_model = os.getenv("BASE_MODEL", "gpt-4.1-mini")
        ft_job = client.fine_tuning.jobs.create(
            model=base_model,
            training_file=train_uploaded.id,
            validation_file=val_uploaded.id,
            suffix=cfg.name,
        )

        final_job = wait_for_job(client, ft_job.id)
        tuned_model = final_job.get("fine_tuned_model")

        samples = load_eval_samples(eval_file)
        baseline_model = os.getenv("BASELINE_MODEL", base_model)
        baseline_eval = evaluate_model(client, baseline_model, samples)
        tuned_eval = evaluate_model(client, tuned_model, samples) if tuned_model else {"avg_overlap_score": 0, "samples": []}

    result = {
        "experiment": cfg.name,
        "dataset_version": cfg.dataset_version,
        "description": cfg.description,
        "base_model": os.getenv("BASE_MODEL", "gpt-4.1-mini"),
        "fine_tune_job": final_job,
        "baseline_eval": baseline_eval,
        "tuned_eval": tuned_eval,
        "delta": round(tuned_eval.get("avg_overlap_score", 0) - baseline_eval.get("avg_overlap_score", 0), 4),
    }

    out_dir = Path("artifacts/experiments") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    results = []
    for cfg_path in [Path("configs/experiment_set_a.json"), Path("configs/experiment_set_b.json")]:
        print(f"Running experiment: {cfg_path.name}")
        results.append(run_experiment(cfg_path))

    Path("artifacts/experiments/summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
