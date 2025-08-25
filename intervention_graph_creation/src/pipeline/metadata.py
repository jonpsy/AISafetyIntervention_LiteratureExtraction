from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class Metadata(BaseModel):
    """Metadata class to store publication information for nodes and edges."""

    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    date_published: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    abstract: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    def __str__(self) -> str:
        """String representation showing key metadata."""
        return f"Metadata(paper_id='{self.paper_id}', title='{self.title}', source_type='{self.source_type}')"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_dict(cls, metadata_dict: dict) -> "Metadata":
        """Create Metadata instance from dictionary (compatible with add_metadata.py output)."""
        return cls(**metadata_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return self.model_dump(exclude_none=True)
