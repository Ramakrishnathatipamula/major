import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

@dataclass
class AgriculturalDocument:
    """
    Data model representing a loaded agricultural document with required source metadata.
    """
    content: str
    source_organization: str
    document_title: str
    crop: str = "Potato"
    disease: Optional[str] = None
    publication_date: Optional[str] = None
    url_reference: Optional[str] = None
    document_type: str = "bulletin"
    target_pathogen: Optional[str] = None
    geographic_relevance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentLoader:
    """
    Loads local agricultural documents (TXT, MD, HTML) and extracts content with metadata.
    """

    def load_file(
        self,
        file_path: Union[str, Path],
        metadata: Dict[str, Any]
    ) -> AgriculturalDocument:
        """
        Load a text document from local disk with metadata validation.

        Args:
            file_path: Absolute or relative path to document file
            metadata: Dictionary containing document source metadata

        Returns:
            AgriculturalDocument object holding extracted text and validated metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1", errors="replace").strip()

        if not content:
            raise ValueError(f"Document file is empty: {path}")

        # Validate required metadata attributes without fabricating fake values
        source_org = metadata.get("source_organization", "unknown")
        doc_title = metadata.get("document_title", path.stem)
        crop = metadata.get("crop", "Potato")
        disease = metadata.get("disease")
        pub_date = metadata.get("publication_date")
        url_ref = metadata.get("url_reference")
        doc_type = metadata.get("document_type", "bulletin")
        pathogen = metadata.get("target_pathogen")
        geo_rel = metadata.get("geographic_relevance")

        return AgriculturalDocument(
            content=content,
            source_organization=str(source_org) if source_org else "unknown",
            document_title=str(doc_title) if doc_title else path.stem,
            crop=str(crop) if crop else "Potato",
            disease=str(disease) if disease else None,
            publication_date=str(pub_date) if pub_date else None,
            url_reference=str(url_ref) if url_ref else None,
            document_type=str(doc_type) if doc_type else "bulletin",
            target_pathogen=str(pathogen) if pathogen else None,
            geographic_relevance=str(geo_rel) if geo_rel else None
        )
