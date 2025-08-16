import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.document_parser import DocumentMetadata


class MetadataRepository:
    """Simple repository that persists DocumentMetadata objects as JSON files on disk."""

    def __init__(self, root_dir: str | Path = "metadata_store"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, metadata: DocumentMetadata) -> Path:
        """Persist a DocumentMetadata instance to a JSON file.

        The filename is a UUID plus the original filename to avoid collisions.
        Returns the path to the saved JSON file.
        """
        file_id = uuid.uuid4().hex
        safe_name = Path(metadata.filename).name.replace(" ", "_")
        json_path = self.root_dir / f"{file_id}_{safe_name}.json"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(metadata), f, ensure_ascii=False, indent=2)

        return json_path

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _to_dict(self, metadata: DocumentMetadata) -> dict[str, Any]:
        """Serialize DocumentMetadata to a JSON-serialisable dict."""
        return {
            "filename": metadata.filename,
            "file_type": metadata.file_type,
            "total_pages": metadata.total_pages,
            "total_chars": metadata.total_chars,
            "total_tokens": metadata.total_tokens,
            "sections": metadata.sections,
            "saved_at": datetime.utcnow().isoformat(),
        }
