from dataclasses import dataclass
from enum import Enum
import numpy as np
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


class DecisionReason(str, Enum):
    MATCHED = "matched"
    MATCHED_RERANKER = "matched_reranker"
    EMPTY_INDEX = "empty_index"
    NO_CANDIDATES = "no_candidates"
    BELOW_THRESHOLD = "below_threshold"
    AMBIGUOUS_MARGIN = "ambiguous_margin"
    RERANKER_REJECTED = "reranker_rejected"


class RouteCandidate(BaseModel):
    """One unique route candidate and its retrieval evidence."""

    model_config = ConfigDict(frozen=True)

    route_name: str
    score: float
    threshold: float
    passed_threshold: bool


class RouterResult(BaseModel):
    """Observable routing decision returned by ``match`` and ``amatch``."""

    model_config = ConfigDict(frozen=True)

    route: Optional[Route] = None
    score: Optional[float] = None
    margin: Optional[float] = None
    candidates: List[RouteCandidate] = Field(default_factory=list)
    decision_reason: DecisionReason

    @property
    def matched(self) -> bool:
        return self.route is not None

    @property
    def route_name(self) -> Optional[str]:
        return self.route.name if self.route is not None else None


@dataclass
class RollbackSnapshot:
    route: Optional['Route'] = None
    embeddings: Optional[np.ndarray] = None
