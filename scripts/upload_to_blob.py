from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


def upload_folder(local_folder: Path, container_name: str, prefix: str = "") -> None:
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    service = BlobServiceClient.from_connection_string(connection_string)
    container = service.get_container_client(container_name)

    try:
        container.create_container()
    except Exception:
        pass

    for path in sorted(local_folder.rglob("*.jsonl")):
        blob_name = f"{prefix}{path.as_posix()}"
        with path.open("rb") as data:
            container.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"Uploaded: {path} -> {container_name}/{blob_name}")


if __name__ == "__main__":
    container = os.getenv("AZURE_STORAGE_CONTAINER", "kla-finetune")
    upload_folder(Path("data/sets"), container)
