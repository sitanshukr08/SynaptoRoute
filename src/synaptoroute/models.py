import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, StringConstraints, ConfigDict
from typing_extensions import Annotated

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class Route(BaseModel):
    """
    Represents a single semantic route or intent.
    """
    model_config = ConfigDict(validate_assignment=True)
    name: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    utterances: List[NonEmptyString] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=-1.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata_serializable(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata must be JSON serializable: {e}")
        return v

    @field_validator('utterances')
    @classmethod
    def deduplicate_utterances(cls, v: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for utt in v:
            if utt not in seen:
                seen.add(utt)
                deduped.append(utt)
        if not deduped:
            raise ValueError("Route must have at least one valid utterance.")
        return deduped

from dataclasses import dataclass
import numpy as np

@dataclass
class RollbackSnapshot:
    route: Optional['Route'] = None
    embeddings: Optional[np.ndarray] = None
